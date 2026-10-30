#!/usr/bin/env python3
"""Associate NVTX dispatch descriptors with graph launches and GPU envelopes."""
import argparse,bisect,collections,json,sqlite3,statistics
p=argparse.ArgumentParser();p.add_argument('database');p.add_argument('output');a=p.parse_args();c=sqlite3.connect('file:'+a.database+'?mode=ro',uri=True)
markers=collections.defaultdict(list)
for start,tid,plain,interned in c.execute("select start,globalTid,text,s.value from NVTX_EVENTS n left join StringIds s on n.textId=s.id where n.text like 'qwen_v2|%' or s.value like 'qwen_v2|%'"):
 text=plain or interned;fields=text.split('|');markers[tid].append((start,fields[1:]))
for values in markers.values():values.sort()
kernels=collections.defaultdict(list)
for row in c.execute('select correlationId,start,end,graphId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null'):kernels[row[0]].append(row)
records=[]
for start,end,tid,corr in c.execute("select r.start,r.end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like '%GraphLaunch%'"):
 if corr not in kernels or tid not in markers:continue
 ms=markers[tid];i=bisect.bisect_right([x[0] for x in ms],start)-1
 if i<0:continue
 fields=ms[i][1];rs=kernels[corr];first=min(r[1] for r in rs);last=max(r[2] for r in rs)
 records.append(dict(phase=fields[0],kind=fields[1],actual_rows=int(fields[2]),padded_rows=int(fields[3]),requests=fields[4],graph_id=rs[0][3],gpu_kernel_envelope_us=(last-first)/1000,host_graph_api_us=(end-start)/1000,kernel_executions=len(rs),correlation_id=corr))
groups=collections.defaultdict(list)
for r in records:groups[(r['phase'],r['kind'],r['actual_rows'],r['padded_rows'],r['graph_id'])].append(r)
summary=[dict(phase=k[0],kind=k[1],actual_rows=k[2],padded_rows=k[3],graph_id=k[4],replays=len(rs),median_kernel_envelope_us=statistics.median(r['gpu_kernel_envelope_us'] for r in rs),median_host_graph_api_us=statistics.median(r['host_graph_api_us'] for r in rs),kernel_executions_per_replay=statistics.median(r['kernel_executions'] for r in rs)) for k,rs in sorted(groups.items())]
open(a.output,'w').write(json.dumps({'scope':'Profiler-instrumented graph replay, mapped to host NVTX dispatch markers. GPU kernel envelopes run from first kernel start to last kernel end, excluding leading/trailing non-kernel nodes. They are NOT additive critical-path contributions. Draft fused-step envelopes can include multiple draft passes. No hardware-counter estimates are inferred.','summary':summary,'records':records},indent=2))
print('Mapped',len(records),'graph launches')
