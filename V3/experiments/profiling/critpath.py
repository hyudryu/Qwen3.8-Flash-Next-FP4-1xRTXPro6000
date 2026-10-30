#!/usr/bin/env python3
"""Critical-path-ish attribution: split one graph replay's timeline into elementary slices; each slice is
charged to the active kernel(s) split equally; idle slices are 'gap'. Sums to the replay span."""
import collections, sqlite3, statistics, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from analyze import cat
c = sqlite3.connect(sys.argv[1]); gid = int(sys.argv[2]); n = int(sys.argv[3])
rows = c.execute("""select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k
  join StringIds s on k.demangledName=s.id where k.graphId=? order by k.start""", (gid,)).fetchall()
reps = [rows[i:i + n] for i in range(0, len(rows) - n + 1, n)]
tot = collections.defaultdict(list)
for rep in reps[1:]:
    pts = sorted({p for r in rep for p in (r[0], r[1])})
    acc = collections.defaultdict(float)
    for a, b in zip(pts, pts[1:]):
        act = [r for r in rep if r[0] <= a and r[1] >= b]
        if not act:
            acc['gap'] += (b - a) / 1e3
        for r in act:
            acc[cat(r[2], (r[3], r[4], r[5]))] += (b - a) / 1e3 / len(act)
    for k, v in acc.items():
        tot[k].append(v)
span = sum(statistics.median(v) for v in tot.values())
print(f'replays={len(reps) - 1} attributed span~{span:.0f}us')
for k, v in sorted(tot.items(), key=lambda x: -statistics.median(x[1]))[:16]:
    print(f'  {statistics.median(v):8.1f}us {100 * statistics.median(v) / span:5.1f}%  {k}')
