#!/usr/bin/env python3
"""Attribute per-replay kernel time for each CUDA graph in a node-traced nsys sqlite."""
import collections, sqlite3, statistics, sys


def cat(nm, g):
    if 'MoEDynamicKernelSilu' in nm or 'csf_Pair' in nm or 'route_plan' in nm or 'compact' in nm or 'MetadataPrefetch' in nm:
        return 'moe-routed'
    if 'W4A16FusedMoe' in nm:
        return 'moe-w4a16(mtp)'
    if 'gemvx' in nm or 'sorted_softmax_topk' in nm or 'sigmoid' in nm.lower():
        return 'router'
    if 'o640111100' in nm or 'act_and_mul' in nm:
        return 'shared-expert'
    if 'hyperconnection' in nm or 'gate_mean' in nm or 'hc_up' in nm or 'splitKreduce' in nm:
        return 'hc'
    if 'nvjet' in nm and g in ((8, 1, 16), (1, 80, 1)):
        return 'hc'
    if '(1, 2, 3880)' in str(g) or g == (31040, 1, 1) or g[2] == 3880 or g[1] == 3880:
        return 'lm-head'
    if 'o6144111100' in nm:
        return 'out-proj'
    if 'o2560111100' in nm and g == (1, 1, 256):
        return 'gdn-in-proj'
    if 'o2560111100' in nm:
        return 'proj-other-2560'
    if 'gdn' in nm or 'conv1d' in nm or 'gated_rmsnorm' in nm or 'Recurrent' in nm:
        return 'gdn-core'
    if any(t in nm.lower() for t in ('atte', 'qsa', 'topk', 'reshape_and_cache', 'rope', 'representatives', 'compress', 'ring')):
        return 'attn-core'
    if 'reduceWeightOnly' in nm:
        return 'splitk-reduce'
    if 'FillFunctor' in nm:
        return 'fill'
    if 'direct_copy' in nm or 'elementwise' in nm:
        return 'torch-elementwise'
    return 'other:' + nm.replace('kernel_cutlass_kernel_', '')[:50]


def main(path):
    c = sqlite3.connect(path)
    rows = c.execute("""select k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.graphId from CUPTI_ACTIVITY_KIND_KERNEL k
                        join StringIds s on k.demangledName=s.id order by k.start""").fetchall()
    graphs = collections.defaultdict(list)
    for r in rows:
        graphs[r[6]].append(r)
    for gid, ks in sorted(graphs.items(), key=lambda x: -sum(r[1] - r[0] for r in x[1]))[:4]:
        # split into replays by gaps > 300us
        reps, cur = [], [ks[0]]
        for r in ks[1:]:
            if r[0] - max(x[1] for x in cur[-5:]) > 300000:
                reps.append(cur); cur = []
            cur.append(r)
        reps.append(cur)
        size = statistics.mode(len(x) for x in reps)
        reps = [x for x in reps if len(x) == size]
        spans = [(max(r[1] for r in x) - x[0][0]) / 1e3 for x in reps]
        print(f'graph {gid}: replays={len(reps)} kernels={size} median span={statistics.median(spans):.1f}us')
        agg = collections.defaultdict(lambda: [0, []])
        for x in reps:
            per = collections.defaultdict(float)
            for r in x:
                k = cat(r[2], (r[3], r[4], r[5])); per[k] += (r[1] - r[0]) / 1e3
            for k, v in per.items():
                agg[k][1].append(v)
        for r in reps[0]:
            agg[cat(r[2], (r[3], r[4], r[5]))][0] += 1
        for k, v in sorted(agg.items(), key=lambda x: -statistics.median(x[1][1]))[:18]:
            print(f'  {statistics.median(v[1]):8.1f}us n={v[0]:4d} {k}')


if __name__ == '__main__':
    main(sys.argv[1])
