"""Independent GPU parity/graph checks; no weights or serving process required."""
import torch,triton
from kernels.hc_fused import hc_up_gate
from kernels.csf_routes import compact_experts
from b12x.norm.hyperconnection._kernels import _gate_mean_launch
torch.manual_seed(42)
for n in (1,4):
 x=torch.randn(n,320,device='cuda',dtype=torch.bfloat16)*.1
 w=torch.randn(10240,320,device='cuda',dtype=torch.bfloat16)*.1
 normalized=torch.randn(n,10240,device='cuda',dtype=torch.bfloat16)
 reference=torch.empty(n,2560,device='cuda',dtype=torch.bfloat16);actual=torch.empty_like(reference)
 _gate_mean_launch(normalized,x@w.T,reference,4,2560,256)
 hc_up_gate[((n+15)//16,160)](x,w,normalized,actual,n,2560,320,4,16,num_warps=4)
 error=(actual.float()-reference.float()).abs().max().item();assert error<=.008,error
 print('HC rows',n,'max_abs_error',error)
for dtype in (torch.int32,torch.int64):
 for n in (10,40,80,160,320,640):
  ids=torch.randint(-10,530,(n,),device='cuda',dtype=dtype)
  if dtype==torch.int64:ids[0]=2**40
  output=torch.empty(min(n,512),device='cuda',dtype=torch.int32)
  def run():compact_experts[(1,)](ids,output,n,512,triton.next_power_of_2(n))
  run();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):run()
  g.replay();torch.cuda.synchronize()
  valid=ids[(ids>=0)&(ids<512)].unique(sorted=True).to(torch.int32)
  expected=torch.full_like(output,-1);expected[:len(valid)]=valid
  assert torch.equal(output,expected),(dtype,n)
print('Passed 12 exact route-compaction/graph cases; original authoritative IDs retained')
