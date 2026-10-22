"""Reference planner for MegaMoE dynamic expert load balancing.

Pure Python, no torch: the device planner in ``smxx_mega_moe_eplb.cuh``
implements the same algorithm and the tests compare the two, so this file is
the specification of the plan.

Vocabulary: ``R`` ranks, ``E`` logical experts, ``H = E / R`` resident experts per
rank, ``S`` redundant slots per rank. Rank ``r`` owns logical experts
``[r * H, (r + 1) * H)`` in its local slots ``[0, H)``; redundant slots are local
slots ``[H, H + S)``. The physical slot id of local slot ``j`` on rank ``r`` is
``r * (H + S) + j``.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class Plan:
    num_ranks: int
    num_experts: int
    num_redundant_slots: int
    block_m: int
    # blocks[e][r]: token blocks of expert e executed on rank r
    blocks: List[List[int]]
    # slot_expert[r][s]: logical expert held by redundant slot s of rank r, or -1
    slot_expert: List[List[int]]
    # copies[e]: (rank, cumulative blocks) in assignment order, resident rank first
    copies: List[List[Tuple[int, int]]]
    loads: List[int] = field(default_factory=list)

    @property
    def resident_experts(self) -> int:
        return self.num_experts // self.num_ranks

    @property
    def slots_per_rank(self) -> int:
        return self.resident_experts + self.num_redundant_slots

    def resident_rank(self, expert: int) -> int:
        return expert // self.resident_experts

    def local_slot(self, rank: int, expert: int) -> int:
        if self.resident_rank(expert) == rank:
            return expert - rank * self.resident_experts
        return self.resident_experts + self.slot_expert[rank].index(expert)

    def physical_slot(self, rank: int, expert: int) -> int:
        return rank * self.slots_per_rank + self.local_slot(rank, expert)

    def copy_list(self, rank: int, prev_slot_expert: Optional[Sequence[int]]) -> List[Tuple[int, int]]:
        """(redundant slot, expert) pairs rank ``rank`` must fetch from the resident_rank."""
        out = []
        for s, e in enumerate(self.slot_expert[rank]):
            if e >= 0 and (prev_slot_expert is None or prev_slot_expert[s] != e):
                out.append((s, e))
        return out


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def plan_reference(counts: Sequence[Sequence[int]],
                   num_redundant_slots: int,
                   block_m: int,
                   prev_slot_expert: Optional[Sequence[Sequence[int]]] = None,
                   open_gate_blocks: int = 0,
                   max_rounds: Optional[int] = None) -> Plan:
    """Load-only block-granular replica planning (MegaFlux, App. A.1).

    ``counts[r][e]`` is the number of tokens rank ``r`` routes to expert ``e``.
    ``prev_slot_expert[r][s]`` is the expert a redundant slot currently holds;
    reusing it costs no weight copy, so such ranks are preferred receivers
    and are exempt from ``open_gate_blocks``, the minimum allocation that
    justifies copying weights into a fresh slot.
    """
    R = len(counts)
    E = len(counts[0])
    S = num_redundant_slots
    assert E % R == 0 and block_m > 0 and S >= 0
    H = E // R
    resident_rank = [e // H for e in range(E)]
    if prev_slot_expert is None:
        prev_slot_expert = [[-1] * S for _ in range(R)]
    cached = [set(x for x in prev_slot_expert[r] if x >= 0) for r in range(R)]

    total = [sum(counts[r][e] for r in range(R)) for e in range(E)]
    b = [_ceil_div(total[e], block_m) for e in range(E)]
    q = [[0] * R for _ in range(E)]
    for e in range(E):
        q[e][resident_rank[e]] = b[e]
    load = [sum(q[e][r] for e in range(E)) for r in range(R)]
    tau = _ceil_div(sum(b), R)

    def replicas_on(r: int) -> int:
        return sum(1 for e in range(E) if resident_rank[e] != r and q[e][r] > 0)

    def holds(r: int, e: int) -> bool:
        return q[e][r] > 0 or e in cached[r]

    rounds = (S + 2) if max_rounds is None else max_rounds
    for _ in range(rounds):
        moved = False
        donors = sorted((r for r in range(R) if load[r] > tau),
                        key=lambda r: (-(load[r] - tau), r))
        for d in donors:
            experts = sorted((e for e in range(d * H, (d + 1) * H) if q[e][d] > 0),
                             key=lambda e: (-q[e][d], e))
            for e in experts:
                if load[d] <= tau:
                    break
                # Existing copies first (active or cached), least loaded first.
                receivers = sorted((r for r in range(R) if r != d and holds(r, e)),
                                   key=lambda r: (load[r], r))
                chosen = None
                for r in receivers:
                    v = min(q[e][d], max(0, tau - load[r]))
                    if v > 0 and (q[e][r] > 0 or replicas_on(r) < S):
                        chosen = (r, v)
                        break
                if chosen is None:
                    fresh = sorted((r for r in range(R)
                                    if r != d and not holds(r, e) and load[r] < tau
                                    and replicas_on(r) < S),
                                   key=lambda r: (load[r], r))
                    for r in fresh:
                        v = min(q[e][d], max(0, tau - load[r]))
                        if v >= max(1, open_gate_blocks):
                            chosen = (r, v)
                            break
                if chosen is None:
                    continue
                r, v = chosen
                q[e][d] -= v
                q[e][r] += v
                load[d] -= v
                load[r] += v
                moved = True
        if not moved:
            break

    # Refinement 1: remove replicas whose allocation fits elsewhere under the
    # current maximum load; prefer the resident rank, then the least loaded copy.
    max_load = max(load)
    replicas = sorted(((q[e][r], e, r) for e in range(E) for r in range(R)
                       if resident_rank[e] != r and q[e][r] > 0))
    for v, e, r in replicas:
        if q[e][r] != v or v == 0:
            continue
        targets = [resident_rank[e]] + sorted((t for t in range(R)
                                       if t != r and t != resident_rank[e] and q[e][t] > 0),
                                      key=lambda t: (load[t], t))
        for t in targets:
            if load[t] + v <= max_load:
                q[e][t] += v
                q[e][r] = 0
                load[t] += v
                load[r] -= v
                break

    # Refinement 2: single-block moves between the extremes while they share
    # an expert and differ by more than one block. Never opens a copy.
    for _ in range(E * R):
        hi = max(range(R), key=lambda r: (load[r], -r))
        lo = min(range(R), key=lambda r: (load[r], r))
        if load[hi] - load[lo] <= 1:
            break
        shared = [e for e in range(E) if q[e][hi] > 0 and q[e][lo] > 0]
        if not shared:
            break
        e = shared[0]
        q[e][hi] -= 1
        q[e][lo] += 1
        load[hi] -= 1
        load[lo] += 1

    # Slot assignment: keep cached slots still in use, then fill free slots.
    slot_expert = [list(prev_slot_expert[r]) for r in range(R)]
    for r in range(R):
        needed = [e for e in range(E) if resident_rank[e] != r and q[e][r] > 0]
        assert len(needed) <= S, "replica capacity exceeded"
        keep = set(needed)
        for s in range(S):
            if slot_expert[r][s] not in keep:
                slot_expert[r][s] = -1
        free = [s for s in range(S) if slot_expert[r][s] < 0]
        for e in needed:
            if e not in slot_expert[r]:
                slot_expert[r][free.pop(0)] = e

    copies = []
    for e in range(E):
        order = [resident_rank[e]] + [r for r in range(R) if r != resident_rank[e] and q[e][r] > 0]
        cum, acc = [], 0
        for r in order:
            acc += q[e][r]
            cum.append((r, acc))
        copies.append(cum)

    return Plan(R, E, S, block_m, q, slot_expert, copies, load)


def remap_reference(plan: Plan, counts: Sequence[Sequence[int]],
                    topk_idx_per_rank: Sequence[Sequence[Sequence[int]]]) -> List[List[List[int]]]:
    """Map every rank's logical ``topk_idx`` to physical slot ids (the
    logical-to-physical remap of the top-k ids).

    Tokens of one expert are ordered by source rank, then by their position in
    that rank's routing table; the block index of that ordinal selects the copy.
    Negative (masked) ids pass through.
    """
    R, E = plan.num_ranks, plan.num_experts
    base = [[sum(counts[rr][e] for rr in range(r)) for e in range(E)] for r in range(R)]
    out = []
    for r in range(R):
        seen = [0] * E
        rows = []
        for row in topk_idx_per_rank[r]:
            new_row = []
            for e in row:
                if e < 0:
                    new_row.append(e)
                    continue
                ordinal = base[r][e] + seen[e]
                seen[e] += 1
                block = ordinal // plan.block_m
                dest = plan.copies[e][-1][0]
                for rank, cum in plan.copies[e]:
                    if block < cum:
                        dest = rank
                        break
                new_row.append(plan.physical_slot(dest, e))
            rows.append(new_row)
        out.append(rows)
    return out


def _expert_totals(counts) -> List[int]:
    """Per-expert totals from either a flat histogram ``counts[e]`` or the
    planner's per-rank table ``counts[r][e]`` (summed over ranks)."""
    if hasattr(counts, 'tolist'):
        counts = counts.tolist()
    if len(counts) and hasattr(counts[0], '__len__'):
        E = len(counts[0])
        return [int(sum(row[e] for row in counts)) for e in range(E)]
    return [int(x) for x in counts]


def static_placement(counts, num_ranks: int, block_m: int = 64) -> List[int]:
    """Static expert placement: which logical expert each physical resident slot holds.

    The kernel's contiguous placement is contiguous (physical slot ``p`` is resident
    slot ``p % H`` of rank ``p // H``), so relabeling the experts chooses
    which rank owns which expert at no per-call cost: the weights are loaded
    in permuted order and the routing ids pass through a logical-to-physical
    table. This picks the relabeling that balances the per-rank block cost of
    a calibration histogram, the prefill objective of :func:`plan_reference`
    with no redundant slots: ``cost[e] = ceil(total[e] / block_m)``, minimise the
    busiest rank's sum with exactly ``H = E / num_ranks`` experts per rank.

    ``counts`` is the calibration histogram, flat ``counts[e]`` or per rank
    ``counts[r][e]`` (summed). EPLB-style greedy first: experts by descending
    block cost (ascending id on ties), each to the least loaded rank with a
    free resident slot (lowest rank on ties). Then pairwise swap refinement: while
    some swap of one expert of the busiest rank with one expert of another
    rank lowers the pair's higher load, the swap bringing the two closest
    together is taken (up to ``2 * E`` swaps). The same refinement runs from
    the identity, and the better of the two (lower busiest-rank load, then
    lower sum of squared loads, then the identity) is returned, so the result
    is never worse than the contiguous placement on the calibration
    histogram, and an all-equal histogram keeps the identity. Deterministic,
    pure Python, so every rank derives the same table.

    Returns ``perm`` with ``perm[p]`` the logical expert in physical slot
    ``p`` (``len(perm) == E``, every rank gets exactly ``H`` experts, sorted
    by logical id within a rank). ``perm.index(e)`` is the physical slot of
    logical expert ``e``; the kernel's resident rank of ``e`` becomes
    ``perm.index(e) // H``.
    """
    total = _expert_totals(counts)
    E = len(total)
    R = num_ranks
    assert R > 0 and E % R == 0 and block_m > 0
    H = E // R
    b = [_ceil_div(t, block_m) for t in total]

    def refine(groups: List[List[int]]) -> None:
        load = [sum(b[e] for e in g) for g in groups]
        for _ in range(2 * E):
            hi = max(range(R), key=lambda r: (load[r], -r))
            best = None  # (new pair max, -reduction tie, lo, i, j)
            for lo in range(R):
                if lo == hi or load[lo] >= load[hi]:
                    continue
                gap = load[hi] - load[lo]
                for i, a in enumerate(groups[hi]):
                    for j, c in enumerate(groups[lo]):
                        d = b[a] - b[c]
                        if d <= 0 or d >= gap:
                            continue
                        key = (max(load[hi] - d, load[lo] + d), -d, lo, i, j)
                        if best is None or key < best:
                            best = key
            if best is None or best[0] >= load[hi]:
                break
            _, d, lo, i, j = best
            d = -d
            groups[hi][i], groups[lo][j] = groups[lo][j], groups[hi][i]
            load[hi] -= d
            load[lo] += d

    def score(groups: List[List[int]]):
        load = [sum(b[e] for e in g) for g in groups]
        return (max(load), sum(x * x for x in load))

    # EPLB-style greedy (longest processing time first) under the slot limit.
    greedy: List[List[int]] = [[] for _ in range(R)]
    load = [0] * R
    for e in sorted(range(E), key=lambda x: (-b[x], x)):
        r = min((r for r in range(R) if len(greedy[r]) < H), key=lambda r: (load[r], r))
        greedy[r].append(e)
        load[r] += b[e]
    refine(greedy)
    identity: List[List[int]] = [list(range(r * H, (r + 1) * H)) for r in range(R)]
    refine(identity)
    groups = greedy if score(greedy) < score(identity) else identity
    return [e for g in groups for e in sorted(g)]


def apply_static_placement(counts, perm: Sequence[int]):
    """Reorder a histogram into physical order: entry ``p`` of the result is
    the count of logical expert ``perm[p]``. Accepts a flat ``counts[e]`` or
    the planner's per-rank ``counts[r][e]`` (each row is reordered), so
    :func:`plan_reference` and :func:`static_redundant_slots` run unchanged
    on the permuted layout (their physical slot ids then index ``perm``)."""
    if hasattr(counts, 'tolist'):
        counts = counts.tolist()
    if len(counts) and hasattr(counts[0], '__len__'):
        return [[row[e] for e in perm] for row in counts]
    return [counts[e] for e in perm]


def static_redundant_slots(load: Sequence[float], num_ranks: int, num_redundant_slots: int) -> List[List[int]]:
    """Fill the redundant slots once from a load history (EPLB-style replication).

    ``load[e]`` is the historical load of logical expert ``e`` (token count);
    experts are owned contiguously (``resident_rank = e // H``). Repeatedly the
    expert with the largest load per copy gets one more copy on the least
    loaded rank that has a free redundant slot and does not hold it yet, with
    the load split evenly over the copies. Deterministic, so every rank
    derives the same table. Returns ``slot_expert[rank][slot]`` (-1 = empty).
    """
    E = len(load)
    H = E // num_ranks
    copies = [1] * E
    holders = [{e // H} for e in range(E)]
    rank_load = [0.0] * num_ranks
    for e in range(E):
        rank_load[e // H] += float(load[e])
    slots = [[-1] * num_redundant_slots for _ in range(num_ranks)]
    free = [num_redundant_slots] * num_ranks
    for _ in range(num_ranks * num_redundant_slots):
        order = sorted(range(E), key=lambda x: (-float(load[x]) / copies[x], x))
        placed = False
        for e in order:
            if float(load[e]) <= 0:
                break
            share_old = float(load[e]) / copies[e]
            share_new = float(load[e]) / (copies[e] + 1)
            candidates = [r for r in range(num_ranks) if free[r] > 0 and r not in holders[e]]
            if not candidates:
                continue
            r = min(candidates, key=lambda x: (rank_load[x], x))
            # Worth a slot only if the new rank stays below the holders' level
            # and the holders shed a visible share of the mean rank load.
            if rank_load[r] + share_new >= max(rank_load[h] for h in holders[e]) \
                    or share_old - share_new < 0.01 * sum(rank_load) / num_ranks:
                continue
            for h in holders[e]:
                rank_load[h] -= share_old - share_new
            rank_load[r] += share_new
            holders[e].add(r)
            copies[e] += 1
            slots[r][num_redundant_slots - free[r]] = e
            free[r] -= 1
            placed = True
            break
        if not placed:
            break
    return slots


def _fill_by_frequency(freq: Sequence[float], num_ranks: int, num_redundant_slots: int) -> List[List[int]]:
    """EPLB-style replication on activation frequency without the "worth a
    slot" gates of ``static_redundant_slots``: every redundant slot is filled.
    Repeatedly the expert with the largest frequency per copy gets one more
    copy on the least loaded rank that has a free slot and does not hold it."""
    E = len(freq)
    H = E // num_ranks
    S = num_redundant_slots
    copies = [1] * E
    holders = [{e // H} for e in range(E)]
    rank_load = [0.0] * num_ranks
    for e in range(E):
        rank_load[e // H] += float(freq[e])
    slots = [[-1] * S for _ in range(num_ranks)]
    free = [S] * num_ranks
    for _ in range(num_ranks * S):
        order = sorted(range(E), key=lambda x: (-float(freq[x]) / copies[x], x))
        for e in order:
            candidates = [r for r in range(num_ranks) if free[r] > 0 and r not in holders[e]]
            if not candidates:
                continue
            share_old = float(freq[e]) / copies[e]
            share_new = float(freq[e]) / (copies[e] + 1)
            r = min(candidates, key=lambda x: (rank_load[x], x))
            for h in holders[e]:
                rank_load[h] -= share_old - share_new
            rank_load[r] += share_new
            holders[e].add(r)
            copies[e] += 1
            slots[r][S - free[r]] = e
            free[r] -= 1
            break
        else:
            break
    return slots
