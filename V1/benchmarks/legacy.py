#!/usr/bin/env python3
"""Dependency-free vLLM streaming benchmark. No restart or serving mutations."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import urllib.request


def request(base, path, body=None, timeout=1800):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('BENCH_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['BENCH_API_KEY']
    req = urllib.request.Request(base.rstrip('/') + path,
                                 data=None if body is None else json.dumps(body).encode(), headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(base, path, body=None):
    with request(base, path, body) as response:
        return json.load(response)


def sse(response):
    """Parse SSE events, including multiple data lines and CRLF."""
    data = []
    for raw in response:
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                value = '\n'.join(data)
                if value == '[DONE]':
                    return
                yield json.loads(value)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data and '\n'.join(data) != '[DONE]':
        yield json.loads('\n'.join(data))


def summarize_stream(events, start, end, expected_input, expected_output):
    arrivals, text, usage, finish = [], [], None, None
    for timestamp, event in events:
        if event.get('usage'):
            usage = event['usage']
        for choice in event.get('choices', []):
            content = choice.get('text', '')
            if content:
                arrivals.append(timestamp)
                text.append(content)
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
    gaps = [(b-a)*1000 for a, b in zip(arrivals, arrivals[1:])]
    n = usage.get('completion_tokens') if usage else None
    p = usage.get('prompt_tokens') if usage else None
    cached = ((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
    ttft = arrivals[0]-start if arrivals else None
    # Includes time to receive all output after the first chunk. Chunk multiplicity
    # is unknown: this is an apparent TPOT estimate, never kernel/token ITL.
    decode = arrivals[-1]-arrivals[0] if len(arrivals) > 1 else None
    valid = bool(n == expected_output and p == expected_input and arrivals and cached == 0)
    return dict(status='ok' if valid else 'invalid', prompt_tokens=p, output_tokens=n,
                cached_tokens=cached, cache_verified=cached == 0,
                finish_reason=finish, elapsed_s=end-start,
                ttft_ms=ttft*1000 if ttft is not None else None,
                apparent_tpot_ms=decode*1000/(n-1) if decode is not None and n and n > 1 else None,
                apparent_decode_tps=(n-1)/decode if decode and n and n > 1 else None,
                output_tps=n/(end-start) if n is not None else None,
                chunk_gap_p50_ms=statistics.median(gaps) if gaps else None,
                chunk_gap_p99_ms=sorted(gaps)[min(len(gaps)-1, int(len(gaps)*.99))] if gaps else None,
                content_chunks=len(arrivals), usage=usage,
                output_sha256=hashlib.sha256(''.join(text).encode()).hexdigest(),
                chunk_arrivals_s=[x-start for x in arrivals],
                invalid_reason=None if valid else 'length mismatch, missing content/usage, or cold prefix cache not verified')


def completion(base, model, prompt, output, seed, barrier=None):
    body = dict(model=model, prompt=prompt, max_tokens=output,
                temperature=0, seed=seed, ignore_eos=True, stream=True,
                stream_options={'include_usage': True}, n=1)
    if barrier is not None:
        barrier.wait()
    start = time.perf_counter()
    events = []
    try:
        with request(base, '/v1/completions', body) as response:
            for event in sse(response):
                events.append((time.perf_counter(), event))
        return summarize_stream(events, start, time.perf_counter(), len(prompt), output)
    except Exception as exc:
        return {'status': 'error', 'error': str(exc), 'elapsed_s': time.perf_counter()-start}


def make_prompt(base, model, length, seed):
    # Tokenize meaningful prose. Randomize the FIRST block (not a suffix) to avoid
    # cross-request prefix-cache hits. Exact IDs are stored for paired reruns.
    rng = random.Random(seed)
    nonce = ' '.join(str(rng.randrange(100000, 999999)) for _ in range(96))
    passage = ('Analyze the following engineering notes. Explain the design, identify '
               'failure modes, and propose tests with a detailed implementation. '
               'The service receives requests, checks a cache, runs inference, and '
               'streams results. Consider concurrency, latency, memory, and correctness. ')
    text = nonce + '\n' + passage * (length//20+100)
    tokens = get_json(base, '/tokenize', {'model': model, 'prompt': text, 'add_special_tokens': False})['tokens']
    if len(tokens) < length:
        raise ValueError('Insufficient tokenized prompt length')
    return tokens[:length]


def metrics(base):
    try:
        with request(base, '/metrics', timeout=10) as r:
            return r.read().decode()
    except Exception as exc:
        return '# unavailable: ' + str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='nvidia/Qwen3.8-Flash-Next-FP4-Local')
    parser.add_argument('--label', required=True, help='Actual externally configured server variant')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True, help='Shared persisted paired workload JSON')
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 4, 8])
    parser.add_argument('--input-lengths', nargs='+', type=int, default=[1024, 8192, 32768])
    parser.add_argument('--output-lengths', nargs='+', type=int, default=[512, 1024])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--prefill-only', action='store_true', help='One output token: TTFT proxy, not pure GPU prefill')
    parser.add_argument('--cold-prefix-confirmed', action='store_true',
                        help='Operator verified prefix caching is disabled or reset before this run')
    args = parser.parse_args()
    if min(args.concurrency+args.input_lengths+args.output_lengths+[args.repeats]) < 1 or args.warmups < 1:
        parser.error('Positive dimensions and at least one warmup required')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = vars(args).copy()
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}
    metadata['started_unix'] = time.time()
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    try:
        models = get_json(args.base_url, '/v1/models')
        (args.output/'models.json').write_text(json.dumps(models, indent=2))
        if args.model not in [m['id'] for m in models['data']]:
            raise ValueError('Requested model absent from /v1/models; use actual served alias')
        workload = json.loads(args.workload.read_text()) if args.workload.exists() else {}
        outputs = [1] if args.prefill_only else args.output_lengths
        for c in args.concurrency:
            for length in args.input_lengths:
                for output in outputs:
                    for repeat in range(-args.warmups, args.repeats):
                        for i in range(c):
                            key = f'{c}/{length}/{output}/{repeat}/{i}'
                            if key not in workload:
                                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')
                                workload[key] = make_prompt(args.base_url, args.model, length, seed)
        args.workload.parent.mkdir(parents=True, exist_ok=True)
        args.workload.write_text(json.dumps(workload))
        (args.output/'workload.sha256').write_text(hashlib.sha256(args.workload.read_bytes()).hexdigest())
        rows, groups = [], []
        import threading
        for c in args.concurrency:
            for length in args.input_lengths:
                for output in outputs:
                    for repeat in range(-args.warmups, args.repeats):
                        before = metrics(args.base_url)
                        barrier = threading.Barrier(c+1)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                            futures = [pool.submit(completion, args.base_url, args.model,
                                workload[f'{c}/{length}/{output}/{repeat}/{i}'], output, 42, barrier) for i in range(c)]
                            start = time.perf_counter()
                            barrier.wait()
                            results = [f.result() for f in futures]
                            elapsed = time.perf_counter()-start
                        tag = f'c{c}-i{length}-o{output}-r{repeat}'
                        (args.output/(tag+'-metrics-before.txt')).write_text(before)
                        (args.output/(tag+'-metrics-after.txt')).write_text(metrics(args.base_url))
                        for i, row in enumerate(results):
                            if args.cold_prefix_confirmed and row.get('cached_tokens') is None and row.get('status') == 'invalid':
                                if row.get('prompt_tokens') == length and row.get('output_tokens') == output and row.get('content_chunks'):
                                    row.update(status='ok', invalid_reason=None, cache_verified='operator-confirmed')
                            row.update(label=args.label, concurrency=c, input_length=length,
                                       output_length=output, repeat=repeat, request=i, warmup=repeat < 0)
                            rows.append(row)
                        valid = all(r['status'] == 'ok' for r in results)
                        group = dict(label=args.label, concurrency=c, input_length=length,
                                     output_length=output, repeat=repeat, warmup=repeat < 0,
                                     status='ok' if valid else 'invalid', wall_s=elapsed,
                                     aggregate_output_tps=sum(r.get('output_tokens') or 0 for r in results)/elapsed if valid else None)
                        groups.append(group)
                        with (args.output/'requests.jsonl').open('a') as f:
                            for row in results:
                                f.write(json.dumps(row)+'\n')
                        (args.output/'groups.json').write_text(json.dumps(groups, indent=2))
                        print(json.dumps(group), flush=True)
        with (args.output/'requests.csv').open('w', newline='') as f:
            fields = [k for k in rows[0] if k not in ('usage','chunk_arrivals_s')]
            writer = csv.DictWriter(f, fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        if any(g['status'] != 'ok' for g in groups):
            raise RuntimeError('Invalid measurements present; inspect raw results, do not compare as valid throughput')
    except Exception as exc:
        (args.output/'failure.json').write_text(json.dumps({'status':'blocked_or_failed', 'error':str(exc)}, indent=2))
        raise SystemExit(str(exc))


if __name__ == '__main__':
    main()
