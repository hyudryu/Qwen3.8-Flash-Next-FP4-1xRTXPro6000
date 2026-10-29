"""Tokenize code/prose calibration sources with the model tokenizer (run inside the container)."""
import collections, glob, json, os, random
from tokenizers import Tokenizer

SNAP = glob.glob('/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/*/tokenizer.json')[0]
tok = Tokenizer.from_file(SNAP)
SRC = '/opt/v3/draft_vocab/corpus_src'  # collected TS/JS/Markdown/C/Rust text files (not shipped)
out = {}


def count(name, texts):
    c = collections.Counter()
    total = 0
    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) == 256:
            for e in tok.encode_batch(batch, add_special_tokens=False):
                c.update(e.ids); total += len(e.ids)
            batch = []
    for e in tok.encode_batch(batch, add_special_tokens=False):
        c.update(e.ids); total += len(e.ids)
    out[name] = {'total': total, 'counts': c}
    print(name, total, len(c), flush=True)


def chunks(path, size=4000):
    text = open(path, errors='ignore').read()
    return [text[i:i + size] for i in range(0, len(text), size)]


files = glob.glob('/usr/lib/python3.12/**/*.py', recursive=True)
random.Random(1).shuffle(files)
py, n = [], 0
for f in files:
    t = open(f, errors='ignore').read()
    py += [t[i:i + 4000] for i in range(0, len(t), 4000)]
    n += len(t)
    if n > 30e6:
        break
count('python', py)
for name in ('ts', 'js', 'md', 'c', 'rs'):
    count(name, chunks(f'{SRC}/{name}.txt'))
json.dump(out, open('/opt/v3-out/code_counts.json', 'w'))
