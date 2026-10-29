#!/usr/bin/env python3
"""C1 coding benchmark (V1 coding.py, natural EOS, unique cache salts) plus MTP acceptance counters."""
import argparse, hashlib, json, pathlib, subprocess, sys, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repository root
CODING = ROOT / 'V1/benchmarks/coding.py'


def spec(port):
    text = urllib.request.urlopen(f'http://localhost:{port}/metrics', timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        for key in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens'):
            if line.startswith(f'vllm:spec_decode_{key}_total{{'):
                out[key] = out.get(key, 0) + float(line.rsplit(' ', 1)[1])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--rounds', type=int, default=5)
    a = p.parse_args()
    out = ROOT / 'V3/results' / a.label
    before = spec(a.port)
    subprocess.run([sys.executable, str(CODING), '--output', str(out), '--base-url', f'http://localhost:{a.port}',
                    '--metrics-port', str(a.port), '--concurrency', '1', '--rounds', str(a.rounds)],
                   check=True, stdout=subprocess.DEVNULL)
    after = spec(a.port)
    d = {k: after[k] - before.get(k, 0) for k in after}
    d['accepted_per_draft'] = d['num_accepted_tokens'] / max(d['num_drafts'], 1)
    s = json.loads((out / 'summary.json').read_text())['summaries'][0]
    reqs = json.loads((out / 'requests.json').read_text())
    res = {'label': a.label, 'median_tps': s['median_aggregate_output_tps'], 'round_tps': s['round_tps'],
           'median_decode_tps': s['median_apparent_decode_tps'], 'spec': d,
           'output_sha256': [hashlib.sha256(r['output'].encode()).hexdigest()[:16] for r in reqs]}
    (out / 'v3-summary.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('label', 'median_tps', 'round_tps', 'median_decode_tps')}),
          'accepted/draft=%.3f' % d['accepted_per_draft'])


if __name__ == '__main__':
    main()
