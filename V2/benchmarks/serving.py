#!/usr/bin/env python3
"""Natural-EOS serving benchmark. Counts committed usage, never SSE chunks."""
import argparse, atexit, concurrent.futures, hashlib, json, pathlib, statistics, random, subprocess, threading, time, urllib.request

def call(base,path,body=None,timeout=1800):
    req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(req,timeout=timeout)

def fetch(base,path,body=None,timeout=1800):
    with call(base,path,body,timeout=timeout) as r:return json.load(r)

def events(response):
    data=[]
    for raw in response:
        line=raw.decode().rstrip('\r\n')
        if not line and data:
            value='\n'.join(data);data=[]
            if value=='[DONE]':return
            yield json.loads(value)
        elif line.startswith('data:'):data.append(line[5:].lstrip())

def completion(base,model,prompt,limit,seed,barrier=None):
    body=dict(model=model,prompt=prompt,max_tokens=limit,temperature=0,seed=seed,stream=True,
              stream_options={'include_usage':True,'continuous_usage_stats':True})
    if barrier:barrier.wait()
    start=time.perf_counter();arrivals=[];text=[];finish=None;usage=None
    with call(base,'/v1/completions',body) as response:
        for e in events(response):
            now=time.perf_counter()
            if e.get('usage'):
                usage=e['usage'];n=usage['completion_tokens']
                if n and (not arrivals or n>arrivals[-1][1]):arrivals.append((now-start,n))
            for c in e.get('choices',[]):
                text.append(c.get('text',''));finish=c.get('finish_reason') or finish
    wall=time.perf_counter()-start
    if not usage:raise RuntimeError('Missing committed-token usage')
    first,last=(arrivals[0],arrivals[-1]) if arrivals else ((None,0),(None,0))
    decode=(last[1]-first[1])/(last[0]-first[0]) if len(arrivals)>1 and last[0]>first[0] else None
    return dict(start_monotonic=start,wall_s=wall,ttft_s=first[0],committed_tokens=usage['completion_tokens'],
                request_tokens_s=usage['completion_tokens']/wall,decode_tokens_s=decode,
                decode_window_tokens=last[1]-first[1],decode_window_s=last[0]-first[0] if arrivals else None,
                finish_reason=finish,capped=finish=='length',usage=usage,
                committed_arrivals=arrivals,output_sha256=hashlib.sha256(''.join(text).encode()).hexdigest())

def workload(base,model,path,length):
    if path.exists():
        data=json.loads(path.read_text())
        if any(len(row['tokens'])!=length for row in data):raise ValueError('Persisted workload length differs; use a fresh workload path')
        if not all('content_offset' in row for row in data):raise ValueError('Legacy raw-completion workload: use a fresh chat-template workload file')
        return data
    def template_ids(text):
        return fetch(base,'/tokenize',dict(model=model,messages=[{'role':'user','content':text}],add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))['tokens']
    ta,tb=template_ids('A'),template_ids('B')
    front=0
    while front<min(len(ta),len(tb)) and ta[front]==tb[front]:front+=1
    back=0
    while back<min(len(ta),len(tb))-front and ta[-back-1]==tb[-back-1]:back+=1
    header=ta[:front];suffix=ta[-back:] if back else []
    if not header or not suffix:raise ValueError('Unable to isolate chat generation template')
    content_length=length-len(header)-len(suffix)
    tasks=[
      'Implement a bounded asynchronous job queue in Python, with cancellation, retries, shutdown, and executable tests. Explain race conditions.',
      'Implement a Rust LRU cache with constant-time operations and tests. Explain ownership and eviction invariants.',
      'Write a design review of a database migration service. Include rollback, idempotency and failure injection.',
      'Design JSON tool schemas for a deployment assistant, then give example tool requests and responses for a failed rollout.'
    ]
    notes='Review notes: preserve correctness under cancellation. Each transaction must be idempotent. Test stale reads, timeouts, retries, and concurrent updates.\n'
    result=[]
    for i in range(16):
        task=tasks[i%len(tasks)]
        prefix=fetch(base,'/tokenize',dict(model=model,prompt=f'Case {i}:\n'+notes*250,add_special_tokens=False))['tokens']
        tail=fetch(base,'/tokenize',dict(model=model,prompt='\nTask: '+task+'\nAnswer:\n',add_special_tokens=False))['tokens']
        result.append({'case':i,'task':task,'content_offset':len(header),'template_suffix_tokens':len(suffix),'tokens':header+prefix[:content_length-len(tail)]+tail+suffix})
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result));return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',default='http://localhost:8001');p.add_argument('--model',default='qwen3.8-flash-next');p.add_argument('--variant',required=True);p.add_argument('--workload',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--concurrency',type=int,nargs='+',default=[1,8]);p.add_argument('--repeat-start',type=int,default=0);p.add_argument('--repeats',type=int,default=5);p.add_argument('--input',type=int,default=2048);p.add_argument('--output-tokens',type=int,default=1024);p.add_argument('--tracks',nargs='+',choices=['cold','warm'],default=['cold','warm']);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    telemetry=(a.output/'gpu0-telemetry.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--id=0','--query-gpu=timestamp,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,memory.used,utilization.gpu','--format=csv','--loop-ms=1000'],stdout=telemetry)
    atexit.register(monitor.terminate)
    work=workload(a.base,a.model,a.workload,a.input)
    prefix_path=a.workload.with_name(a.workload.stem+'-paired-prefixes.json')
    prefixes=json.loads(prefix_path.read_text()) if prefix_path.exists() else {}
    def paired_prompt(c,track,repeat,j):
        key=f'{c}:{track}:{repeat}:{j}'+(':'+a.variant if repeat<0 else '')
        if key not in prefixes:
            rng=random.Random(int(hashlib.sha256(key.encode()).hexdigest(),16))
            text='Workload '+key+'\n'+' '.join(str(rng.randrange(100000,999999)) for _ in range(96))
            prefixes[key]=fetch(a.base,'/tokenize',dict(model=a.model,prompt=text,add_special_tokens=False))['tokens'][:64]
        case=work[(repeat+j)%len(work)];offset=case['content_offset']
        return case['tokens'][:offset]+prefixes[key]+case['tokens'][offset+64:]

    for c in a.concurrency:
      for track in a.tracks:
        for local_repeat in range(-1,a.repeats):
          repeat=local_repeat+a.repeat_start if local_repeat>=0 else -1
          prompts=[paired_prompt(c,track,repeat,j) for j in range(c)]
          prefix_path.write_text(json.dumps(prefixes))
          if track=='warm':
            for prompt in prompts:completion(a.base,a.model,prompt,1,0)
          barrier=threading.Barrier(c)
          with call(a.base,'/metrics') as r:before=r.read().decode()
          start=time.perf_counter()
          with concurrent.futures.ThreadPoolExecutor(c) as pool:
            futures=[pool.submit(completion,a.base,a.model,prompt,a.output_tokens,j,barrier) for j,prompt in enumerate(prompts)]
            samples=[f.result() for f in futures]
          wall=time.perf_counter()-start
          if track=='cold' and any((s['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') != 0 for s in samples):
            raise RuntimeError('Cold-prefix definition violated: cached prompt tokens')
          with call(a.base,'/metrics') as r:after=r.read().decode()
          common_start=max(s['start_monotonic']+s['committed_arrivals'][0][0] for s in samples if s['committed_arrivals'])
          common_end=min(s['start_monotonic']+s['committed_arrivals'][-1][0] for s in samples if s['committed_arrivals'])
          def count_at(stream,t):
            return max([n for dt,n in stream['committed_arrivals'] if stream['start_monotonic']+dt<=t] or [0])
          window_tokens=sum(count_at(s,common_end)-count_at(s,common_start) for s in samples) if common_end>common_start else 0
          steady=window_tokens/(common_end-common_start) if common_end>common_start else None
          record=dict(steady_state_aggregate_tokens_s=steady,steady_state_window_tokens=window_tokens,steady_state_window_s=max(0,common_end-common_start),variant=a.variant,concurrency=c,track=track,repeat=repeat,warmup=repeat<0,wall_s=wall,
                      aggregate_committed_tokens_s=sum(s['committed_tokens'] for s in samples)/wall,
                      streams=samples,scope='Natural EOS with output cap; cold requests include prefill. Decode windows use cumulative committed usage counts.')
          stem=f'{track}-c{c}-r{repeat}'
          (a.output/(stem+'.json')).write_text(json.dumps(record,indent=2))
          (a.output/(stem+'-metrics-before.txt')).write_text(before);(a.output/(stem+'-metrics-after.txt')).write_text(after)
          print(json.dumps({k:v for k,v in record.items() if k not in ['streams','scope']}),flush=True)
if __name__=='__main__':main()
