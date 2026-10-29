#!/usr/bin/env python3
"""Pick draft-vocab subsets from corpus counts; report held-out coverage.

Held-out sets: output texts (JSON lists of {"output": ...}); not used for selection.
Usage: select_vocab.py HELDOUT_JSON [HELDOUT_JSON ...]
"""
import json, pathlib, sys, urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]  # draft_vocab/
COUNTS = HERE / 'counts'
VOCAB = 248320
SPECIAL_FROM = 248044  # bos/eos and all added control tokens live at the top of the vocab
# Source mixture: generated chat/prose/code + real code and docs.
WEIGHTS = {'generated': 0.25, 'multilingual': 0.30, 'python': 0.12, 'ts': 0.08, 'js': 0.05, 'md': 0.08, 'c': 0.05,
           'rs': 0.07}


def tokenize(text):
    r = urllib.request.Request('http://localhost:8010/tokenize', data=json.dumps(
        {'model': 'qwen3.8-flash-next', 'prompt': text, 'add_special_tokens': False}).encode(),
        headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())['tokens']


def main():
    data = json.loads((COUNTS / 'corpus_counts.json').read_text())
    sources = {'generated': data, 'multilingual': json.loads((COUNTS / 'corpus_ml_counts.json').read_text()),
               **json.loads((COUNTS / 'code_counts.json').read_text())}
    score = {}
    for name, src in sources.items():
        w = WEIGHTS[name] / src['total']
        for k, v in src['counts'].items():
            score[int(k)] = score.get(int(k), 0.) + w * v
    counts = score
    ranked = sorted(counts, key=lambda t: (-counts[t], t))
    special = list(range(SPECIAL_FROM, VOCAB))
    helds = {pathlib.Path(f).parent.name or f: [t for r in json.loads(pathlib.Path(f).read_text()) for t in tokenize(r['output'])]
             for f in sys.argv[1:]}
    print(f"sources={ {n: s['total'] for n, s in sources.items()} } distinct={len(counts)}")
    for k in (32768, 49152, 65536, 81920, 98304):
        chosen = list(dict.fromkeys(special + ranked))[:k]
        filler = (t for t in range(VOCAB) if t not in set(chosen))
        chosen += [next(filler) for _ in range(k - len(chosen))]
        s = set(chosen)
        (HERE / f'top{k}.json').write_text(json.dumps({'k': k, 'ids': sorted(chosen),
                                                        'source': 'weighted mix ' + json.dumps(WEIGHTS) + ' + special tokens'}))
        print(f'K={k:6d} ' + '  '.join(f'{name}: {100 * sum(t in s for t in held) / len(held):.2f}%'
                                         for name, held in helds.items()))


if __name__ == '__main__':
    main()
