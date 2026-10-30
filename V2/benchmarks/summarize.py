#!/usr/bin/env python3
"""Paired throughput estimates, confidence intervals, and request/MTP statistics."""
import argparse,collections,json,math,pathlib,random,re,statistics

def percentile(values,q):
    values=sorted(values)
    if not values:return None
    position=(len(values)-1)*q;lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)

def geometric(values):return math.exp(statistics.mean(math.log(x) for x in values)) if values else None

def counter(path,name):
    total=0.
    for line in path.read_text().splitlines():
        if re.match(re.escape(name)+r'(?:\{|\s)',line):total+=float(line.rsplit(' ',1)[1])
    return total

def load(directory,variant):
    result=[]
    for path in directory.glob('*.json'):
        r=json.loads(path.read_text())
        if r.get('warmup') is not False:continue
        r['comparison_variant']=variant;r['source_file']=str(path)
        before=path.with_name(path.stem+'-metrics-before.txt');after=path.with_name(path.stem+'-metrics-after.txt')
        names={'draft_rounds':'vllm:spec_decode_num_drafts_total','draft_tokens':'vllm:spec_decode_num_draft_tokens_total','accepted_drafts':'vllm:spec_decode_num_accepted_tokens_total','counter_generation_tokens':'vllm:generation_tokens_total','counter_requests':'vllm:request_success_total'}
        r['counter_deltas']={key:counter(after,name)-counter(before,name) for key,name in names.items()}
        r['counter_tokens_match']=r['counter_deltas']['counter_generation_tokens']==sum(s['committed_tokens'] for s in r['streams'])
        result.append(r)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--require-pairs',type=int,default=20);a=p.parse_args()
    records=[]
    for folder,variant in [('v1-a','v1'),('v1-a2','v1'),('v2-b1','v2'),('v2-b2','v2'),('original-chat-control','original'),('target-only-diagnostic','target-only')]:
        if (a.root/folder).exists():records+=load(a.root/folder,variant)
    grouped=collections.defaultdict(list)
    for r in records:grouped[(r['comparison_variant'],r['concurrency'],r['track'])].append(r)
    summaries=[]
    for (variant,c,track),rs in sorted(grouped.items()):
        streams=[s for r in rs for s in r['streams']];deltas={k:sum(r['counter_deltas'][k] for r in rs) for k in rs[0]['counter_deltas']};rounds=deltas['draft_rounds']
        summaries.append(dict(variant=variant,concurrency=c,track=track,batches=len(rs),requests=len(streams),request_aggregate_tokens_s_geomean=geometric([r['aggregate_committed_tokens_s'] for r in rs]),steady_aggregate_tokens_s_geomean=geometric([r['steady_state_aggregate_tokens_s'] for r in rs if r['steady_state_aggregate_tokens_s']]),ttft_p50_ms=percentile([s['ttft_s']*1000 for s in streams if s['ttft_s'] is not None],.5),ttft_p95_ms=percentile([s['ttft_s']*1000 for s in streams if s['ttft_s'] is not None],.95),request_latency_p50_s=percentile([s['wall_s'] for s in streams],.5),request_latency_p95_s=percentile([s['wall_s'] for s in streams],.95),stream_decode_ms_per_token_p50=percentile([1000/s['decode_tokens_s'] for s in streams if s['decode_tokens_s']],.5),committed_tokens=sum(s['committed_tokens'] for s in streams),capped_fraction=sum(s['capped'] for s in streams)/len(streams),counter_token_match_batches=sum(r['counter_tokens_match'] for r in rs),accepted_drafts_per_request_round=deltas['accepted_drafts']/rounds if rounds else None,committed_tokens_per_request_round=(sum(s['committed_tokens'] for s in streams)-len(streams))/rounds if rounds else None,counter_deltas=deltas))
    paired=[];rng=random.Random(61109)
    for c in [1,8]:
      for track in ['cold','warm']:
        left={r['repeat']:r for r in grouped.get(('v1',c,track),[])};right={r['repeat']:r for r in grouped.get(('v2',c,track),[])}
        keys=sorted(left.keys()&right.keys())
        if len(keys)<a.require_pairs:raise RuntimeError(f'{c}/{track}: only {len(keys)} matched pairs')
        for metric in ['aggregate_committed_tokens_s','steady_state_aggregate_tokens_s']:
          logs=[math.log(right[i][metric]/left[i][metric]) for i in keys if left[i][metric] and right[i][metric]]
          bootstrap=[statistics.mean(rng.choices(logs,k=len(logs))) for _ in range(10000)]
          paired.append(dict(concurrency=c,track=track,metric=metric,pairs=len(logs),paired_geometric_improvement_pct=(math.exp(statistics.mean(logs))-1)*100,ci95_pct=[(math.exp(percentile(bootstrap,q))-1)*100 for q in [.025,.975]],per_pair_pct=[(math.exp(x)-1)*100 for x in logs]))
    a.output.write_text(json.dumps({'scope':'Main cells use MTP3. Target-only is a separate diagnostic. Geometric throughput ratios use paired batch/repeat indices; 95% percentile bootstrap over pairs (10,000 resamples). Request and steady-window definitions are never mixed.','summaries':summaries,'paired':paired},indent=2))
    for row in paired:print(json.dumps({k:v for k,v in row.items() if k!='per_pair_pct'}))
if __name__=='__main__':main()
