"""A/B harness for the counter-based synchronisation steps of the SM90 NVFP4
fused MegaMoE kernel (8 ranks, one process per GPU).

Modes
  correctness : run the reference knob set (OFF) and every candidate knob set on
                the same inputs; y must be bit-identical (all M, several seeds),
                then a stress loop of alternating calls. Optionally dumps /
                compares y against a directory written by another build.
  bench       : their protocol -- profile-off CUDA events around the kernel, one
                dist.barrier() + L2 flush before every call, 25 same-process
                A/B/B/A blocks (50 calls per arm), per call the maximum latency
                over the 8 ranks, median over the 50 calls.
  stamps      : per-phase globaltimer stamps (kernel slots, see the body), per
                rank, median over calls, for OFF and every candidate knob set.

Knob sets are given as comma-separated env assignments, e.g.
  --arms OFF "DG_NVFP4_PUSH_DISPATCH=1" "DG_NVFP4_PUSH_DISPATCH=1,DG_NVFP4_FINE_COMBINE=1"
OFF means every counter knob forced to 0 (their barrier path).
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm  # noqa: E402
from deep_gemm.quantization_nvfp4 import quantize_to_nvfp4  # noqa: E402
from deep_gemm.utils import per_token_cast_to_fp8  # noqa: E402
from deep_gemm.utils.dist import init_dist  # noqa: E402

SHAPES = {
    'flash': dict(hidden=4096, intermediate_hidden=2048, num_experts=256, num_topk=6),
    'pro': dict(hidden=7168, intermediate_hidden=3072, num_experts=384, num_topk=6),
    'mimo': dict(hidden=6144, intermediate_hidden=2048, num_experts=384, num_topk=8),
}
COUNTER_KNOBS = ('DG_NVFP4_PUSH_DISPATCH', 'DG_NVFP4_FINE_COMBINE', 'DG_NVFP4_NO_CLEAN_BARRIER',
                 'DG_NVFP4_POOL_STRIDE_DEBUG', 'DG_NVFP4_PUSH_PROXY_FENCE', 'DG_NVFP4_PUSH_GENERIC_ROWS',
                 'DG_NVFP4_PUSH_GPU_SCOPE_DEBUG', 'DG_NVFP4_SYS_TRAFFIC_DEBUG', 'DG_NVFP4_PUSH_STAGGER_NS',
                 'DG_NVFP4_PUSH_CODE_DEBUG')
STAMP_NAMES = {
    0: 'entry(min)', 1: 'barrier1/DONE done', 2: 'pool ready', 3: 'first math(min)',
    4: 'last L1 end', 5: 'last L2 end', 6: 'combine barrier2/first token', 7: 'combine end',
    8: 'routing count done', 9: 'routing writes/pushes issued', 10: 'routing grid sync/DONE signalled',
    11: 'count bcast done', 12: 'barrier3 done',
}
MIN_SLOTS = (0, 3)
ACC_SLOTS = {20: 'L1 task avg us (sum/count)', 22: 'L2 task avg us (sum/count)', 26: 'SM clock over L1 tasks (GHz)',
             24: 'A-loader L1 arrival spin, total us per SM', 25: 'A-loader L2 mask spin, total us per SM'}


def apply_arm(arm: str) -> None:
    for knob in COUNTER_KNOBS:
        os.environ[knob] = '0'
    if arm != 'OFF':
        for kv in arm.split(','):
            k, v = kv.split('=')
            os.environ[k.strip()] = v.strip()


def arm_label(arm: str) -> str:
    if arm == 'OFF':
        return 'OFF'
    return '+'.join((kv.split('=')[0].replace('DG_NVFP4_', '').lower() +
                     ('' if kv.strip().endswith('=1') else kv.split('=')[1]))
                    for kv in arm.split(',') if kv.split('=')[1].strip() != '0')


def make_routing(router: str, m: int, num_experts: int, num_topk: int, rank: int,
                 num_ranks: int, seed: int):
    if router == 'balanced':
        # Every expert receives the same number of rows; a token's slots are distinct.
        rows = torch.arange(m, device='cuda', dtype=torch.int64) + rank * m
        base = rows.unsqueeze(1) * num_topk + torch.arange(num_topk, device='cuda', dtype=torch.int64)
        topk_idx = base % num_experts
        topk_w = torch.full((m, num_topk), 1.0 / num_topk, dtype=torch.float32, device='cuda')
        return topk_idx, topk_w
    # Their bench script's router: torch.manual_seed(rank + seed_offset), random
    # scores, top-k (seed offset 101 in their harness, plus our seed index).
    torch.manual_seed(rank + 101 + seed * 1000)
    scores = torch.randn((m, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    return topk_idx, topk_w.float()


class Case:
    def __init__(self, args, shape, m, rank, num_ranks, group, seed, router):
        self.m = m
        s = SHAPES[shape]
        hidden, ih = s['hidden'], s['intermediate_hidden']
        num_experts, num_topk = s['num_experts'], s['num_topk']
        experts_per_rank = num_experts // num_ranks
        self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, args.cap, num_topk, hidden, ih)
        g = torch.Generator(device='cuda')
        g.manual_seed(seed * 7919 + rank)
        x_bf = torch.randn((m, hidden), dtype=torch.bfloat16, device='cuda', generator=g)
        l1_bf = torch.randn((experts_per_rank, ih * 2, hidden), dtype=torch.bfloat16,
                            device='cuda', generator=g) * 0.05
        l2_bf = torch.randn((experts_per_rank, hidden, ih), dtype=torch.bfloat16,
                            device='cuda', generator=g) * 0.05
        self.topk_idx, self.topk_w = make_routing(router, m, num_experts, num_topk, rank, num_ranks, seed)
        self.x_fp8, self.x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                                      use_packed_ue8m0=False)
        l1_packed, l1_scale = quantize_to_nvfp4(l1_bf, group_size=16)
        l2_packed, l2_scale = quantize_to_nvfp4(l2_bf, group_size=16)
        self.l1, self.l2 = deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
            (l1_packed, l1_scale), (l2_packed, l2_scale), block_n=128)
        self.cum_stats = torch.zeros(experts_per_rank, dtype=torch.int, device='cuda')
        self.y = torch.zeros((m, hidden), dtype=torch.bfloat16, device='cuda')
        self.hidden = hidden
        self.args = args

    def stage(self):
        m = self.m
        self.buffer.x[:m].copy_(self.x_fp8)
        self.buffer.x_sf[:m].copy_(self.x_sf)
        self.buffer.topk_idx[:m].copy_(self.topk_idx)
        self.buffer.topk_weights[:m].copy_(self.topk_w)

    def run(self):
        deep_gemm.nvfp4_mega_moe(
            self.y, self.l1, self.l2, self.buffer,
            cumulative_local_expert_recv_stats=self.cum_stats,
            recipe=(128, 128, 128), activation='swiglu',
            activation_clamp=10.0, fast_math=True,
            kernel_family='fused', family_threshold=256)
        return self.y

    def destroy(self):
        self.buffer.destroy()


def all_ranks_equal(ok: bool, group) -> bool:
    t = torch.tensor([1 if ok else 0], device='cuda', dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def run_correctness(args, rank, num_ranks, group, shape):
    failures = 0
    for m in args.m:
        for seed in range(args.seeds):
            for router in args.routers:
                case = Case(args, shape, m, rank, num_ranks, group, seed, router)
                ys = {}
                for arm in args.arms:
                    apply_arm(arm)
                    case.stage()
                    case.y.zero_()
                    case.run()
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    ys[arm] = case.y.clone()
                ref = ys[args.arms[0]]
                finite = bool(torch.isfinite(ref.float()).all().item())
                line = [f'M={m} seed={seed} router={router} ref={args.arms[0]} finite={finite}']
                ok_all = finite
                for arm in args.arms[1:]:
                    eq = torch.equal(ys[arm].view(torch.int16), ref.view(torch.int16))
                    if not eq:
                        d = (ys[arm].float() - ref.float()).abs()
                        line.append(f'{arm_label(arm)}: MISMATCH max_abs={d.max().item():.3e} n={int((d > 0).sum().item())}')
                    else:
                        line.append(f'{arm_label(arm)}: bit-identical')
                    ok_all = ok_all and eq
                if args.dump_dir:
                    os.makedirs(args.dump_dir, exist_ok=True)
                    torch.save(ref.cpu(), os.path.join(args.dump_dir, f'y_{shape}_M{m}_s{seed}_{router}_r{rank}.pt'))
                if args.compare_dir:
                    path = os.path.join(args.compare_dir, f'y_{shape}_M{m}_s{seed}_{router}_r{rank}.pt')
                    other = torch.load(path).cuda()
                    eq = torch.equal(other.view(torch.int16), ref.view(torch.int16))
                    line.append(f'vs {args.compare_dir}: {"bit-identical" if eq else "MISMATCH"}')
                    ok_all = ok_all and eq
                ok_all = all_ranks_equal(ok_all, group)
                if rank == 0:
                    print(('PASS ' if ok_all else 'FAIL ') + ' | '.join(line), flush=True)
                failures += 0 if ok_all else 1
                # Stress: alternate the arms, every result must equal the reference
                if args.stress > 0 and m == args.m[-1] and seed == 0 and router == args.routers[0]:
                    bad = 0
                    for it in range(args.stress):
                        arm = args.arms[(it % len(args.arms))]
                        apply_arm(arm)
                        case.y.zero_()
                        case.run()
                        if it % 16 == 15:
                            torch.cuda.synchronize()
                            if not torch.equal(case.y.view(torch.int16), ref.view(torch.int16)):
                                bad += 1
                    torch.cuda.synchronize()
                    if not torch.equal(case.y.view(torch.int16), ref.view(torch.int16)):
                        bad += 1
                    ok = all_ranks_equal(bad == 0, group)
                    if rank == 0:
                        print(f'{"PASS" if ok else "FAIL"} stress M={m} calls={args.stress} arms={[arm_label(a) for a in args.arms]} bad_checks={bad}', flush=True)
                    failures += 0 if ok else 1
                case.destroy()
    apply_arm('OFF')
    return failures


def timed_call(case, flush_buf, group, start, end):
    if flush_buf is not None:
        flush_buf.zero_()
    dist.barrier(group=group)
    start.record()
    case.run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0  # us


def run_bench(args, rank, num_ranks, group, shape):
    flush_buf = None
    if args.flush_bytes > 0:
        flush_buf = torch.empty(args.flush_bytes // 4, dtype=torch.int, device='cuda')
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    results = []
    for m in args.m:
        for router in args.routers:
            case = Case(args, shape, m, rank, num_ranks, group, 0, router)
            case.stage()
            # JIT + warm-up of every arm
            for arm in args.arms:
                apply_arm(arm)
                for _ in range(args.warmup):
                    case.run()
                torch.cuda.synchronize()
            dist.barrier(group=group)
            samples = {arm: [] for arm in args.arms}
            order = []
            arms = args.arms
            for _ in range(args.blocks):
                # A/B/B/A over the arm list: A B C ... C B A
                order.extend(arms + arms[::-1])
            for arm in order:
                apply_arm(arm)
                samples[arm].append(timed_call(case, flush_buf, group, start, end))
            case.destroy()
            # gather per-call times across ranks
            row = {'shape': shape, 'm': m, 'router': router, 'calls_per_arm': len(samples[arms[0]]), 'arms': {}}
            for arm in arms:
                local = torch.tensor(samples[arm], dtype=torch.float64, device='cuda')
                gathered = [torch.zeros_like(local) for _ in range(num_ranks)]
                dist.all_gather(gathered, local, group=group)
                per_call = torch.stack(gathered)  # ranks x calls
                max_over_ranks = per_call.max(dim=0).values.cpu().tolist()
                row['arms'][arm_label(arm)] = {
                    'max_rank_median_us': statistics.median(max_over_ranks),
                    'max_rank_mean_us': statistics.mean(max_over_ranks),
                    'max_rank_min_us': min(max_over_ranks),
                    'rank0_median_us': statistics.median(per_call[0].cpu().tolist()),
                    'mean_rank_median_us': statistics.median(per_call.mean(dim=0).cpu().tolist()),
                }
            results.append(row)
            if rank == 0:
                base = row['arms'][arm_label(arms[0])]['max_rank_median_us']
                parts = []
                for arm in arms:
                    r = row['arms'][arm_label(arm)]
                    parts.append(f"{arm_label(arm)}={r['max_rank_median_us']:.1f}us"
                                 + ('' if arm == arms[0] else f" ({(base - r['max_rank_median_us']) / base * 100:+.1f}%)"))
                print(f'BENCH {shape} M={m} router={router} maxrank-median: ' + '  '.join(parts), flush=True)
                print('BENCH_JSON ' + json.dumps(row, sort_keys=True), flush=True)
    apply_arm('OFF')
    return results


def run_stamps(args, rank, num_ranks, group, shape):
    stamps = torch.zeros(32, dtype=torch.int64, device='cuda')
    os.environ['DG_NVFP4_PHASE_STAMPS_PTR'] = str(stamps.data_ptr())
    flush_buf = None
    if args.flush_bytes > 0:
        flush_buf = torch.empty(args.flush_bytes // 4, dtype=torch.int, device='cuda')
    for m in args.m:
        for router in args.routers:
            case = Case(args, shape, m, rank, num_ranks, group, 0, router)
            case.stage()
            for arm in args.arms:
                apply_arm(arm)
                for _ in range(args.warmup):
                    case.run()
                torch.cuda.synchronize()
                dist.barrier(group=group)
                per_call = []
                for _ in range(args.calls):
                    stamps.zero_()
                    for slot in MIN_SLOTS:
                        stamps[slot] = 0x7fffffffffffffff
                    if flush_buf is not None:
                        flush_buf.zero_()
                    dist.barrier(group=group)
                    case.run()
                    torch.cuda.synchronize()
                    s = stamps.cpu().tolist()
                    t0 = s[0]
                    row = [(v - t0) / 1000.0 if (v != 0 and v != 0x7fffffffffffffff) else float('nan')
                           for v in s]
                    # accumulator slots: averages / per-SM totals in us
                    row[20] = s[20] / max(1, s[21]) / 1000.0
                    row[22] = s[22] / max(1, s[23]) / 1000.0
                    row[21] = float(s[21])
                    row[23] = float(s[23])
                    row[26] = s[26] / max(1, s[20])  # GHz over the L1 tasks
                    row[24] = s[24] / 1000.0 / 78.0
                    row[25] = s[25] / 1000.0 / 78.0
                    per_call.append(row)
                # median over calls per slot, per rank; then gather
                med = []
                for slot in range(32):
                    vals = [c[slot] for c in per_call if c[slot] == c[slot]]
                    med.append(statistics.median(vals) if vals else float('nan'))
                local = torch.tensor(med, dtype=torch.float64, device='cuda')
                gathered = [torch.zeros_like(local) for _ in range(num_ranks)]
                dist.all_gather(gathered, local, group=group)
                # entry skew across ranks: gather raw t0 of the last call
                t0s = torch.tensor([float(per_call and stamps.cpu()[0].item())], dtype=torch.float64, device='cuda')
                if rank == 0:
                    tab = torch.stack(gathered)  # ranks x slots
                    print(f'STAMPS {shape} M={m} router={router} arm={arm_label(arm)} (us from kernel entry, median of {args.calls} calls; max over ranks | rank0)', flush=True)
                    for slot, name in STAMP_NAMES.items():
                        col = tab[:, slot]
                        if torch.isnan(col).all():
                            continue
                        print(f'  [{slot:2d}] {name:36s} max={col.nanmax().item() if hasattr(col, "nanmax") else col.max().item():8.2f}  rank0={col[0].item():8.2f}  ranks={[round(v, 1) for v in col.tolist()]}', flush=True)
                    for slot, name in ACC_SLOTS.items():
                        col = tab[:, slot]
                        print(f'  [{slot:2d}] {name:36s} max={col.max().item():8.2f}  rank0={col[0].item():8.2f}  ranks={[round(v, 2) for v in col.tolist()]}', flush=True)
                    print(f'  [21/23] L1/L2 task counts rank0: {int(tab[0, 21].item())} / {int(tab[0, 23].item())}', flush=True)
                    print('STAMPS_JSON ' + json.dumps({'shape': shape, 'm': m, 'router': router, 'arm': arm_label(arm),
                                                       'per_rank_us': tab.tolist()}), flush=True)
            case.destroy()
    del os.environ['DG_NVFP4_PHASE_STAMPS_PTR']
    apply_arm('OFF')


def worker(local_rank, num_local_ranks, args):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(1234 + rank)
    failures = 0
    try:
        for shape in args.shapes:
            if args.mode == 'correctness':
                failures += run_correctness(args, rank, num_ranks, group, shape)
            elif args.mode == 'bench':
                run_bench(args, rank, num_ranks, group, shape)
            elif args.mode == 'stamps':
                run_stamps(args, rank, num_ranks, group, shape)
        dist.barrier(group=group)
        if rank == 0:
            print(f'DONE mode={args.mode} failures={failures}', flush=True)
    finally:
        dist.destroy_process_group()
    if failures:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('correctness', 'bench', 'stamps'), required=True)
    p.add_argument('--shapes', nargs='+', default=['flash', 'pro'], choices=sorted(SHAPES))
    p.add_argument('--m', nargs='+', type=int, default=[1, 2, 8])
    p.add_argument('--cap', type=int, default=8448)
    p.add_argument('--arms', nargs='+', default=['OFF', 'DG_NVFP4_PUSH_DISPATCH=1'])
    p.add_argument('--routers', nargs='+', default=['balanced'], choices=('balanced', 'random'))
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--stress', type=int, default=200)
    p.add_argument('--blocks', type=int, default=25)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--calls', type=int, default=20)
    p.add_argument('--flush-bytes', type=int, default=int(8e9))
    p.add_argument('--dump-dir', default='')
    p.add_argument('--compare-dir', default='')
    p.add_argument('--num-processes', type=int, default=8)
    args = p.parse_args()
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == '__main__':
    main()
