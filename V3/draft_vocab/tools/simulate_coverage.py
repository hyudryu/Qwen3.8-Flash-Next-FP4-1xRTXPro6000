#!/usr/bin/env python3
"""Offline proposal-coverage simulation: static top-K alone vs static + dynamic context tokens.

A target token is covered dynamically if it appeared earlier in the prompt or the output so far.
Held-out sets: coding benchmark requests (prompt+output) and general-check outputs.
"""
import importlib.util, json, pathlib, urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]  # draft_vocab/
V3 = HERE.parent


def tokenize(text):
    r = urllib.request.Request('http://localhost:8010/tokenize', data=json.dumps(
        {'model': 'qwen3.8-flash-next', 'prompt': text, 'add_special_tokens': False}).encode(),
        headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())['tokens']


def main():
    spec = importlib.util.spec_from_file_location('bg', V3 / 'benchmarks/bench_general.py')
    bg = importlib.util.module_from_spec(spec); spec.loader.exec_module(bg)
    general = [(tokenize(p), tokenize(r['output']))
               for p, r in zip(bg.PROMPTS, json.loads((V3 / 'results/general-heldout-outputs.json').read_text()))]
    coding = [(tokenize(r['prompt']), tokenize(r['output']))
              for r in json.loads((V3 / 'results/base-r1/requests.json').read_text())]
    for k in (32768, 49152, 65536, 81920, 98304):
        static = set(json.loads((HERE / f'top{k}.json').read_text())['ids'])
        line = [f'K={k:6d}']
        for name, sets in (('coding', coding), ('general', general)):
            tot = st = dyn = 0
            for prompt, out in sets:
                seen = set(prompt)
                for t in out:
                    tot += 1; st += t in static; dyn += t in static or t in seen
                    seen.add(t)
            line.append(f'{name}: static {100 * st / tot:.2f}% +dyn {100 * dyn / tot:.2f}%')
        print('  '.join(line))


if __name__ == '__main__':
    main()
