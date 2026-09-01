"""Validate symmetric-buffer NVLink all-reduce vs dist.all_reduce for the TP MoE
final reduction (y = sum_r y_r). tp=4 on GPUs 0-3. Correctness + latency."""
import os, sys, time, torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed._symmetric_memory as symm_mem

def worker(rank, world, port, M, H):
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']=str(port)
    dist.init_process_group('nccl', world_size=world, rank=rank,
                            device_id=torch.device(f'cuda:{rank}'))
    torch.cuda.set_device(rank); dev=torch.device('cuda',rank)
    gname = dist.group.WORLD.group_name
    try: symm_mem.enable_symm_mem_for_group(gname)
    except Exception as e:
        if rank==0: print("enable_symm_mem_for_group:", e, flush=True)

    torch.manual_seed(1234+rank)
    yp = torch.randn(M, H, device=dev)                     # this rank's partial FFN output

    # ---- reference: NCCL all-reduce ----
    a = yp.clone(); dist.all_reduce(a, op=dist.ReduceOp.SUM); torch.cuda.synchronize()

    # ---- symmetric-buffer NVLink all-reduce ----
    buf = symm_mem.empty((M, H), dtype=torch.float32, device=dev)
    symm_mem.rendezvous(buf, gname)
    buf.copy_(yp)
    out = torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", gname)
    torch.cuda.synchronize()
    maxdiff = (out - a).abs().max().item()

    # ---- timing (max across ranks) ----
    def bench(fn, it=100):
        for _ in range(10): fn()
        torch.cuda.synchronize(); dist.barrier(); t0=time.time()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.time()-t0)/it*1e3
    def nccl_fn(): b=yp.clone(); dist.all_reduce(b, op=dist.ReduceOp.SUM)
    def symm_fn(): buf.copy_(yp); torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", gname)
    t_nccl = bench(nccl_fn); t_symm = bench(symm_fn)
    tms = torch.tensor([t_nccl, t_symm], device=dev); dist.all_reduce(tms, op=dist.ReduceOp.MAX)

    if rank==0:
        print(f"SYMM_ALLREDUCE maxdiff={maxdiff:.2e} (0=bitexact-ish)  M={M} H={H} tp={world}", flush=True)
        print(f"  NCCL all_reduce   : {tms[0].item():.4f} ms", flush=True)
        print(f"  SYMM one-shot NVLink: {tms[1].item():.4f} ms", flush=True)
        print("SYMM_OK" if maxdiff < 1e-3 else "SYMM_WRONG", flush=True)
    dist.barrier(); os._exit(0)

if __name__=='__main__':
    M=int(os.environ.get('MM','32')); H=6144; world=int(os.environ.get('TP','4'))
    port=int(os.environ.get('MASTER_PORT','21500'))
    mp.spawn(worker, args=(world,port,M,H), nprocs=world, join=True)
