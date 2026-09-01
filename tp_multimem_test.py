"""Validate multimem.red.add (Hopper NVLink multicast) in a load_inline kernel: 4 ranks each
add (rank+1) into a multicast symmetric buffer; after barrier every rank's local buffer == sum = 10.
This is the primitive for fusing FC2's scatter-add + TP all-reduce into one epilogue."""
import os, torch
import torch.distributed as dist, torch.multiprocessing as mp
import torch.distributed._symmetric_memory as symm_mem
from torch.utils.cpp_extension import load_inline
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/tmp/torch_ext_mm'); os.environ['TORCH_CUDA_ARCH_LIST']='9.0'

CUDA=r'''
#include <torch/extension.h>
extern "C" __global__ void mm_add(long long mc, int n, float v){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  unsigned long long a=(unsigned long long)mc + (unsigned long long)i*4;
  asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;"::"l"(a),"f"(v));
}
void run(long long mc,int n,float v){ mm_add<<<(n+255)/256,256>>>(mc,n,v); }
'''
e=load_inline(name='mm_probe',cpp_sources="void run(long long mc,int n,float v);",cuda_sources=CUDA,functions=['run'],extra_cuda_cflags=['-O3'],verbose=False)

def worker(rank, world, port, N):
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']=str(port)
    dist.init_process_group('nccl',world_size=world,rank=rank,device_id=torch.device(f'cuda:{rank}'))
    torch.cuda.set_device(rank); dev=torch.device('cuda',rank)
    gname=dist.group.WORLD.group_name
    try: symm_mem.enable_symm_mem_for_group(gname)
    except Exception: pass
    buf=symm_mem.empty((N,),dtype=torch.float32,device=dev); hdl=symm_mem.rendezvous(buf,gname)
    mc=hdl.multicast_ptr
    if rank==0: print(f"has_multicast_support={hdl.has_multicast_support} multicast_ptr={mc}",flush=True)
    buf.zero_(); dist.barrier(); torch.cuda.synchronize()
    e.run(mc, N, float(rank+1))         # each rank adds (rank+1) into multicast buffer
    torch.cuda.synchronize(); dist.barrier()
    expect=sum(r+1 for r in range(world))   # 1+2+3+4=10
    ok=(buf-expect).abs().max().item()
    if rank==0: print(f"MULTIMEM buf[0]={buf[0].item()} expect={expect} maxerr={ok:.1e}  {'MM_OK' if ok<1e-4 else 'MM_WRONG'}",flush=True)
    dist.barrier(); os._exit(0)

if __name__=='__main__':
    world=int(os.environ.get('TP','4')); port=int(os.environ.get('MASTER_PORT','23700'))
    mp.spawn(worker,args=(world,port,4096),nprocs=world,join=True)
