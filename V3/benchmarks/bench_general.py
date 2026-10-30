#!/usr/bin/env python3
"""C1 non-coding sanity benchmark: prose/math/multilingual/JSON, greedy and sampled.

Reports output tok/s per request (wall incl. prefill) and MTP acceptance. Prompts are distinct from the
draft-vocab calibration prompts and from the coding benchmark.
"""
import argparse, json, pathlib, statistics, time, urllib.request
PROMPTS = [
    'Write a 600-word essay on how lighthouses shaped maritime trade in the 19th century.',
    'Explain, step by step, how to solve the system 3x + 2y = 16 and 5x - y = 7, then verify the answer.',
    'Rédige une lettre formelle à un propriétaire pour signaler une fuite d\'eau dans la salle de bain.',
    '请写一篇关于城市骑行好处的短文，大约500字。',
    'Produce a JSON array of 12 fictional books with title, author, year, genres (array) and a one-line summary.',
    'Describe the plot of an original science-fiction novella about a generation ship whose crew forgets its purpose.',
]


def post(base, body):
    r = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


def spec(base):
    text = urllib.request.urlopen(base + '/metrics', timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        for key in ('num_drafts', 'num_accepted_tokens'):
            if line.startswith(f'vllm:spec_decode_{key}_total{{'):
                out[key] = out.get(key, 0) + float(line.rsplit(' ', 1)[1])
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--label', required=True); p.add_argument('--port', type=int, default=8000)
    a = p.parse_args(); base = f'http://localhost:{a.port}'
    res = {}
    for mode, temp in (('greedy', 0.0), ('sampled', 0.7)):
        post(base, {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': 'warm up'}], 'max_tokens': 32})
        before = spec(base); tps = []
        for i, prompt in enumerate(PROMPTS):
            body = {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 800,
                    'temperature': temp, 'seed': 1234 + i, 'chat_template_kwargs': {'enable_thinking': False},
                    'cache_salt': f'{a.label}-{mode}-{i}-{time.time_ns()}'}
            t = time.perf_counter(); d = post(base, body); el = time.perf_counter() - t
            tps.append(d['usage']['completion_tokens'] / el)
        after = spec(base)
        acc = (after['num_accepted_tokens'] - before['num_accepted_tokens']) / max(after['num_drafts'] - before['num_drafts'], 1)
        res[mode] = {'per_prompt_tps': tps, 'median_tps': statistics.median(tps), 'mean_tps': statistics.mean(tps),
                     'accepted_per_draft': acc}
        print(f'{a.label} {mode}: mean {statistics.mean(tps):.1f} tok/s, median {statistics.median(tps):.1f}, '
              f'accepted/draft {acc:.3f}', flush=True)
    out = pathlib.Path(__file__).resolve().parents[1] / 'results' / f'general-{a.label}.json'
    out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
