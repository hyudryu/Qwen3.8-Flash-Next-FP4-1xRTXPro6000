#!/usr/bin/env python3
"""Summarize graph replay without treating overlapping durations as critical path."""
import argparse,collections,json,sqlite3,statistics

def category(name):
    for needles,label in [(['MoEDynamic','Nvfp4Materialized','W4A16FusedMoe'],'routed_expert_compute'),(['csf_pipeline','nvfp4_csf','CSF'],'expert_scale_preparation'),(['PackedCombineNorm','hyperconnection','hc_up_gate','_gate_mean'],'hyperconnection'),(['topk','TopK','moe_align','route_compact'],'routing'),(['gdn','GDN','RecurrentQwen','causal_conv'],'gdn'),(['Paged','paged','attention','Attention'],'attention'),(['silu_and_mul','swiglu','SiluAndMul'],'ffn_activation_or_fusion'),(['DenseGemm','nvjet','gemm'],'dense_or_shared_projection'),(['ple','Ple','PLE'],'ple')]:
        if any(x in name for x in needles):return label
    return 'other'

def union(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a
        else:total+=max(0,b-end)
        end=b if end is None else max(end,b)
    return total

def main():
    p=argparse.ArgumentParser();p.add_argument('database');p.add_argument('output');a=p.parse_args()
    c=sqlite3.connect('file:'+a.database+'?mode=ro',uri=True)
    rows=list(c.execute('select start,end,s.value,graphId,registersPerThread,staticSharedMemory,dynamicSharedMemory,gridX,gridY,gridZ,blockX,blockY,blockZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName'))
    groups=collections.defaultdict(list)
    for r in rows:groups[r[2]].append(r)
    kernels=[];cats=collections.defaultdict(list)
    for name,rs in groups.items():
        durations=[(r[1]-r[0])/1000 for r in rs];label=category(name);cats[label].extend(rs)
        kernels.append(dict(name=name,category=label,launches=len(rs),sum_ms=sum(durations)/1000,median_us=statistics.median(durations),resources=sorted(set((r[4],r[5]+r[6],r[10]*r[11]*r[12]) for r in rs)),grids=[dict(grid=list(k),count=v) for k,v in collections.Counter(tuple(r[7:10]) for r in rs).items()],graph_launches=sum(r[3] is not None for r in rs)))
    report=dict(scope='Instrumented capture including prefill and warmup. Category sums overlap and are NOT critical-path contributions. Kernel durations are profiler-instrumented, not final serving timing.',kernel_launches=len(rows),kernel_activity_union_ms=union([(r[0],r[1]) for r in rows])/1e6,categories={k:dict(launches=len(rs),summed_kernel_ms=sum(r[1]-r[0] for r in rs)/1e6,activity_union_ms=union([(r[0],r[1]) for r in rs])/1e6) for k,rs in cats.items()},kernels=sorted(kernels,key=lambda k:-k['sum_ms']))
    open(a.output,'w').write(json.dumps(report,indent=2))
if __name__=='__main__':main()
