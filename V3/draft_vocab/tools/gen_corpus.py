#!/usr/bin/env python3
"""Generate a diverse calibration corpus with the served model and count output token ids.

Prompts are deliberately disjoint from the coding benchmark tasks (no TTL cache / work queue /
LRU / worker pool tasks). Outputs are tokenized with the server's /tokenize endpoint.
"""
import collections, concurrent.futures, itertools, json, pathlib, random, sys, urllib.request

BASE = 'http://localhost:8010'
OUT = pathlib.Path(__file__).resolve().parents[1] / 'counts'

LANGS = ['Python', 'TypeScript', 'JavaScript', 'Rust', 'Go', 'C++', 'C', 'Java', 'Kotlin', 'C#', 'Swift', 'Ruby',
         'PHP', 'Haskell', 'Scala', 'Bash', 'SQL', 'Lua', 'Zig', 'Elixir']
CODE_TASKS = [
    'a tokenizer and recursive-descent parser for arithmetic expressions with error recovery',
    'a trie supporting insert, delete and prefix search with unit tests',
    'a command-line tool that parses CSV and prints summary statistics',
    'a rate limiter using the token bucket algorithm',
    'Dijkstra shortest paths on a weighted graph with a binary heap',
    'a JSON pretty printer without using a JSON library',
    'an HTTP request router supporting path parameters',
    'an event emitter with once/on/off semantics and tests',
    'a matrix multiplication routine with blocking and benchmarks',
    'a simple key-value store persisted to an append-only log',
    'a diff algorithm (Myers) for two lists of lines',
    'a bloom filter with configurable false positive rate',
    'a markdown-to-HTML converter supporting headers, lists and code blocks',
    'an interval tree with overlap queries',
    'a retry helper with exponential backoff and jitter',
    'a ring buffer with overwrite and iterator support',
]
PROSE = [
    'Explain how TCP congestion control works, with examples.', 'Summarize the causes of the French Revolution.',
    'Write a short story about a lighthouse keeper who finds a radio.', 'Explain gradient descent to a high school student.',
    'Compare PostgreSQL and MySQL for a startup backend.', 'Write a product requirements document for a habit-tracking app.',
    'Explain the difference between processes and threads.', 'Describe how vaccines train the immune system.',
    'Give a detailed code review checklist for pull requests.', 'Write a cover letter for a junior data analyst role.',
    'Explain how a CPU cache hierarchy works and why it matters.', 'Write meeting notes from a sprint retrospective.',
    'Prove that the square root of 2 is irrational.', 'Solve: a train leaves at 3pm at 60 km/h; another at 4pm at 80 km/h. When does the second catch up?',
    'Explain Bayes theorem with a medical testing example.', 'Derive the formula for the sum of a geometric series.',
    'Explain what a Kubernetes operator is and when to use one.', 'Write a git commit message and PR description for a bug fix in a date parser.',
    'Describe the architecture of a URL shortener service.', 'Explain the CAP theorem with concrete systems as examples.',
]
MULTI = ['Explique en français comment fonctionne la photosynthèse.', '用中文解释什么是机器学习，并举例说明。',
         '한국어로 블록체인의 원리를 설명해 주세요.', 'Explica en español cómo funciona una base de datos relacional.',
         '日本語で再帰関数について説明してください。', 'Erkläre auf Deutsch, wie ein Compiler funktioniert.',
         'Объясни по-русски, что такое хеш-таблица.', 'Explique em português o que é inflação.']
TOOLS = ['Return a JSON object describing three fictional employees with nested address and skills arrays.',
         'Write a YAML GitHub Actions workflow that runs tests for a Node project on three OS targets.',
         'Write a Dockerfile and docker-compose.yml for a Flask app with Postgres and Redis.',
         'Write an OpenAPI 3 spec for a todo-list REST API.', 'Write a Makefile for a C project with tests and clean targets.']


def prompts():
    rnd = random.Random(7)
    out = []
    for lang, task in itertools.product(LANGS, CODE_TASKS):
        if rnd.random() < 0.35:
            out.append((f'Write {task} in {lang}. Include tests and brief explanations.', rnd.random() < 0.3))
    for p in PROSE + MULTI + TOOLS:
        out.append((p, False)); out.append((p, True))
    for task in CODE_TASKS:
        out.append((f'Find and fix the bugs in a typical implementation of {task}; show the corrected code and explain the fixes.', False))
    rnd.shuffle(out)
    return out


def post(path, body):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())


def one(item):
    prompt, think = item
    d = post('/v1/chat/completions', {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': prompt}],
                                      'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 1500,
                                      'chat_template_kwargs': {'enable_thinking': think}})
    m = d['choices'][0]['message']
    text = (m.get('reasoning_content') or m.get('reasoning') or '') + (m.get('content') or '')
    ids = post('/tokenize', {'model': 'qwen3.8-flash-next', 'prompt': text, 'add_special_tokens': False})['tokens']
    return ids


def main():
    items = prompts()
    print(len(items), 'prompts', flush=True)
    counts = collections.Counter(); total = 0
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        for i, ids in enumerate(pool.map(one, items)):
            counts.update(ids); total += len(ids)
            if i % 50 == 0:
                print(i, total, len(counts), flush=True)
    (OUT / 'corpus_counts.json').write_text(json.dumps({'total': total, 'prompts': len(items), 'counts': counts}))
    print('done', total, len(counts))


if __name__ == '__main__':
    main()
