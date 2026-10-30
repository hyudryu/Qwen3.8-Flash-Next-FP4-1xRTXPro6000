#!/usr/bin/env python3
"""Compare shared-output readiness with routed-output readiness at the MoE join."""
import argparse,collections,json,sqlite3,statistics
p=argparse.ArgumentParser();p.add_argument('database');p.add_argument('output');p.add_argument('--graph-id',type=int,default=2540);p.add_argument('--variant',choices=['reference','split'],required=True);a=p.parse_args()
c=sqlite3.connect('file:'+a.database+'?mode=ro',uri=True);groups=collections.defaultdict(lambda:{'shared':[],'routed':[]})
for corr,start,end,name,stream in c.execute('select correlationId,start,end,s.value,streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where graphId=?',(a.graph_id,)):
 if 'MoEDynamicKernelSilu' in name:groups[corr]['routed'].append((start,end,stream))
 shared=name=='split_epilogue' if a.variant=='split' else (name.startswith('void at::native::elementwise_kernel<') and 'BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16,' in name and 'binary_internal::MulFunctor<float>' in name)
 if shared:groups[corr]['shared'].append((start,end,stream))
records=[]
for corr,data in groups.items():
 shared=sorted(data['shared']);routed=sorted(data['routed'])
 if len(shared)!=48 or len(routed)!=48:raise RuntimeError('Unexpected layer count; cannot align chronologically')
 # Decoder layer dependencies serialize successive layer joins. Each layer has
 # one shared-output producer and one routed-output producer in this graph.
 for layer,(s,r) in enumerate(zip(shared,routed)):
  records.append(dict(correlation_id=corr,layer_order=layer,shared_end_ns=s[1],routed_end_ns=r[1],shared_stream=s[2],routed_stream=r[2],shared_ready_first=s[1]<=r[1],shared_late_us=max(0,s[1]-r[1])/1000,readiness_margin_us=(r[1]-s[1])/1000))
result=dict(scope='Instrumented C1 target graph. Pairs are aligned by sequential decoder layer order within each replay. End timestamps establish producer readiness, not a complete event-edge critical-path reconstruction. Host graph API time is not added.',variant=a.variant,layer_instances=len(records),shared_ready_no_later_fraction=sum(r['shared_ready_first'] for r in records)/len(records),median_routed_minus_shared_end_us=statistics.median(r['readiness_margin_us'] for r in records),max_shared_late_us=max(r['shared_late_us'] for r in records),records=records)
open(a.output,'w').write(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
