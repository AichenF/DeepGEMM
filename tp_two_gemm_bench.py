"""Baseline target: DeepGEMM's two standalone grouped GEMMs (FC1 + FC2) for the TP MXFP4
MoE, W4A8 (fp8 x fp4), masked layout, real topk routing distribution. Sum = the SOL target
my fused moe_fp8mma_split kernel should reach."""
import os, sys, torch, random
DG='/home/xutingz/fac/DeepGEMM'
sys.path.insert(0, DG); sys.path.insert(0, DG+'/tests')
import deep_gemm
from generators import generate_m_grouped_masked, QuantConfig
from deep_gemm.testing import bench_kineto

def routing_counts(M,E,topk,seed=0):
    torch.manual_seed(seed); sc=torch.randn(M,E,device='cuda'); _,ti=torch.topk(sc,topk,-1)
    return torch.bincount(ti.reshape(-1),minlength=E).to(torch.int32)

H,I,E,topk,tp = 6144,2048,384,8,4; Is=I//tp
qc=QuantConfig()                                # fp8 x fp8 (W8A8) -- Hopper GEMM floor (fp8_fp4 is Blackwell-only)
recipe,ra,rb=qc.get_recipes()

def bench_gemm(n,k,cnt,tag):
    max_m=max(int(cnt.max().item()),8); exp_m=max(1,int(cnt.float().mean().item()+0.999))
    a,b,mm,psum,d,ref=generate_m_grouped_masked(E,max_m,exp_m,n,k,use_ue8m0=False,quant_config=qc)
    mm.copy_(cnt)
    def fn(): deep_gemm.m_grouped_fp8_gemm_nt_masked(a,b,d,mm,exp_m,disable_ue8m0_cast=True)
    fn(); torch.cuda.synchronize()
    t=bench_kineto(fn,'fp8_gemm',suppress_kineto_output=True)
    if isinstance(t,(list,tuple)): t=sum(t)
    print(f"  {tag}: {t*1e6:7.1f} us  (n={n} k={k} touched={(cnt>0).sum().item()}/{E})",flush=True)
    return t

for M in [int(x) for x in os.environ.get('MS','8 16 32 64 128').split()]:
    cnt=routing_counts(M,E,topk)
    t1=bench_gemm(2*Is,H,cnt,f'M={M} FC1'); t2=bench_gemm(H,Is,cnt,f'M={M} FC2')
    print(f"M={M}: TWO_GEMM (DeepGEMM) = {(t1+t2)*1e3:.3f} ms  [FC1 {t1*1e3:.3f} + FC2 {t2*1e3:.3f}]",flush=True)
