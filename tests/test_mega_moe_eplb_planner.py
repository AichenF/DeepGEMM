"""CPU tests for the reference dynamic-EPLB planner (no GPU, no torch)."""
import importlib.util
import os
import random
import sys
import unittest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'deep_gemm', 'mega', 'eplb_planner.py')
_spec = importlib.util.spec_from_file_location('eplb_planner', _PATH)
eplb_planner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eplb_planner)
plan_reference = eplb_planner.plan_reference
remap_reference = eplb_planner.remap_reference


def _ceil_div(a, b):
    return -(-a // b)


def _baseline_max_load(counts, block_m):
    R, E = len(counts), len(counts[0])
    H = E // R
    loads = [0] * R
    for e in range(E):
        loads[e // H] += _ceil_div(sum(counts[r][e] for r in range(R)), block_m)
    return max(loads)


def _check_invariants(tc, plan, counts, S, prev=None):
    R, E = plan.num_ranks, plan.num_experts
    H = plan.home_experts
    for e in range(E):
        total = sum(counts[r][e] for r in range(R))
        tc.assertEqual(sum(plan.blocks[e]), _ceil_div(total, plan.block_m), f'expert {e} blocks')
        tc.assertEqual(plan.copies[e][0][0], e // H)
        tc.assertEqual(plan.copies[e][-1][1], sum(plan.blocks[e]))
    for r in range(R):
        held = [x for x in plan.slot_expert[r] if x >= 0]
        tc.assertEqual(len(held), len(set(held)))
        tc.assertLessEqual(len(held), S)
        for e in held:
            tc.assertNotEqual(e // H, r, 'owner never needs a helper slot')
        for e in range(E):
            if e // H != r and plan.blocks[e][r] > 0:
                tc.assertIn(e, plan.slot_expert[r])
    tc.assertLessEqual(max(plan.loads), _baseline_max_load(counts, plan.block_m))
    tc.assertEqual(plan.loads, [sum(plan.blocks[e][r] for e in range(E)) for r in range(R)])


def _skewed_counts(R, E, tokens_per_rank, topk, hot_experts, hot_share, seed):
    rng = random.Random(seed)
    counts = [[0] * E for _ in range(R)]
    for r in range(R):
        for _ in range(tokens_per_rank * topk):
            if rng.random() < hot_share:
                e = rng.choice(hot_experts)
            else:
                e = rng.randrange(E)
            counts[r][e] += 1
    return counts


class PlannerTest(unittest.TestCase):
    def test_balanced_routing_creates_no_replicas(self):
        R, E, S = 4, 32, 2
        counts = [[16] * E for _ in range(R)]
        plan = plan_reference(counts, S, 32)
        _check_invariants(self, plan, counts, S)
        self.assertTrue(all(e < 0 for row in plan.slot_expert for e in row))
        self.assertEqual(len(set(plan.loads)), 1)

    def test_hot_expert_is_replicated_and_max_load_drops(self):
        R, E, S = 4, 32, 2
        counts = [[1] * E for _ in range(R)]
        for r in range(R):
            counts[r][5] = 400  # expert 5 on rank 0
        plan = plan_reference(counts, S, 32)
        _check_invariants(self, plan, counts, S)
        self.assertLess(max(plan.loads), _baseline_max_load(counts, 32))
        self.assertGreater(sum(1 for row in plan.slot_expert if 5 in row), 0)
        # Owner stays first and keeps a share.
        self.assertEqual(plan.copies[5][0][0], 0)
        self.assertGreater(plan.blocks[5][0], 0)

    def test_capacity_respected_with_many_hot_experts(self):
        R, E, S = 8, 64, 1
        counts = [[1] * E for _ in range(R)]
        for e in range(0, 8):  # all of rank 0's experts are hot
            for r in range(R):
                counts[r][e] = 64
        plan = plan_reference(counts, S, 16)
        _check_invariants(self, plan, counts, S)
        self.assertLess(max(plan.loads), _baseline_max_load(counts, 16))

    def test_slot_cache_avoids_copies_on_repeat(self):
        R, E, S = 4, 32, 2
        counts = _skewed_counts(R, E, 256, 4, [3, 17], 0.5, seed=1)
        first = plan_reference(counts, S, 32)
        second = plan_reference(counts, S, 32, prev_slot_expert=first.slot_expert)
        self.assertEqual(first.slot_expert, second.slot_expert)
        for r in range(R):
            self.assertEqual(second.copy_list(r, first.slot_expert[r]), [])
            self.assertEqual(len(first.copy_list(r, None)),
                             sum(1 for e in first.slot_expert[r] if e >= 0))

    def test_cached_slot_is_preferred_receiver(self):
        R, E, S = 4, 32, 1
        counts = [[1] * E for _ in range(R)]
        for r in range(R):
            counts[r][5] = 300
        prev = [[-1], [-1], [5], [-1]]
        plan = plan_reference(counts, S, 32, prev_slot_expert=prev)
        _check_invariants(self, plan, counts, S, prev)
        self.assertEqual(plan.slot_expert[2], [5])
        self.assertGreater(plan.blocks[5][2], 0)

    def test_open_gate_blocks_small_replicas(self):
        R, E, S = 4, 32, 2
        counts = [[1] * E for _ in range(R)]
        for r in range(R):
            counts[r][5] = 100
        gated = plan_reference(counts, S, 32, open_gate_blocks=1000)
        _check_invariants(self, gated, counts, S)
        self.assertTrue(all(e < 0 for row in gated.slot_expert for e in row))
        open_ = plan_reference(counts, S, 32, open_gate_blocks=0)
        self.assertLessEqual(max(open_.loads), max(gated.loads))

    def test_determinism(self):
        R, E, S = 8, 96, 3
        counts = _skewed_counts(R, E, 512, 8, [1, 2, 40, 41, 90], 0.4, seed=7)
        a = plan_reference(counts, S, 32)
        b = plan_reference(counts, S, 32)
        self.assertEqual(a.blocks, b.blocks)
        self.assertEqual(a.slot_expert, b.slot_expert)

    def test_remap_matches_plan(self):
        R, E, S, topk, M = 4, 32, 2, 4, 128
        rng = random.Random(3)
        topk_idx = []
        for r in range(R):
            rows = []
            for _ in range(M):
                row = []
                while len(row) < topk:
                    e = rng.choice([3, 17]) if rng.random() < 0.5 else rng.randrange(E)
                    if e not in row:
                        row.append(e)
                rows.append(row)
            topk_idx.append(rows)
        counts = [[0] * E for _ in range(R)]
        for r in range(R):
            for row in topk_idx[r]:
                for e in row:
                    counts[r][e] += 1
        plan = plan_reference(counts, S, 32)
        _check_invariants(self, plan, counts, S)
        phys = remap_reference(plan, counts, topk_idx)
        slots = plan.slots_per_rank
        per_slot = {}
        for r in range(R):
            for row_l, row_p in zip(topk_idx[r], phys[r]):
                for e, p in zip(row_l, row_p):
                    dest_rank, local = divmod(p, slots)
                    if local < plan.home_experts:
                        self.assertEqual(dest_rank * plan.home_experts + local, e)
                    else:
                        self.assertEqual(plan.slot_expert[dest_rank][local - plan.home_experts], e)
                    per_slot[(dest_rank, e)] = per_slot.get((dest_rank, e), 0) + 1
        for e in range(E):
            for r in range(R):
                n = per_slot.get((r, e), 0)
                blocks = plan.blocks[e][r]
                self.assertLessEqual(n, blocks * plan.block_m)
                if blocks > 0:
                    self.assertGreater(n, (blocks - 1) * plan.block_m)
                else:
                    self.assertEqual(n, 0)

    def test_fuzz_invariants(self):
        for seed in range(12):
            rng = random.Random(seed)
            R = rng.choice([2, 4, 8])
            H = rng.choice([4, 8, 12])
            E, S = R * H, rng.choice([1, 2, 3])
            hot = rng.sample(range(E), rng.randint(1, 4))
            counts = _skewed_counts(R, E, rng.choice([16, 128, 1024]), rng.choice([2, 4, 8]),
                                    hot, rng.choice([0.0, 0.3, 0.7]), seed)
            prev = [[rng.choice([-1] + list(range(E))) for _ in range(S)] for _ in range(R)]
            for r in range(R):
                prev[r] = [e if e < 0 or e // H != r else -1 for e in prev[r]]
                seen = set()
                for s in range(S):
                    if prev[r][s] in seen:
                        prev[r][s] = -1
                    seen.add(prev[r][s])
            plan = plan_reference(counts, S, rng.choice([8, 24, 32, 64]), prev_slot_expert=prev,
                                  open_gate_blocks=rng.choice([0, 2, 12]))
            _check_invariants(self, plan, counts, S, prev)


if __name__ == '__main__':
    unittest.main()
