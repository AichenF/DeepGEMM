"""Reference planner for MegaMoE dynamic expert load balancing.

Pure Python, no torch: the device planner in ``smxx_mega_moe_eplb.cuh``
implements the same algorithm and the tests compare the two, so this file is
the specification of the plan.

Vocabulary: ``R`` ranks, ``E`` logical experts, ``H = E / R`` home experts per
rank, ``S`` helper slots per rank. Rank ``r`` owns logical experts
``[r * H, (r + 1) * H)`` in its local slots ``[0, H)``; helper slots are local
slots ``[H, H + S)``. The physical slot id of local slot ``j`` on rank ``r`` is
``r * (H + S) + j``.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class Plan:
    num_ranks: int
    num_experts: int
    num_helper_slots: int
    block_m: int
    # blocks[e][r]: token blocks of expert e executed on rank r
    blocks: List[List[int]]
    # slot_expert[r][s]: logical expert held by helper slot s of rank r, or -1
    slot_expert: List[List[int]]
    # copies[e]: (rank, cumulative blocks) in assignment order, owner first
    copies: List[List[Tuple[int, int]]]
    loads: List[int] = field(default_factory=list)

    @property
    def home_experts(self) -> int:
        return self.num_experts // self.num_ranks

    @property
    def slots_per_rank(self) -> int:
        return self.home_experts + self.num_helper_slots

    def owner(self, expert: int) -> int:
        return expert // self.home_experts

    def local_slot(self, rank: int, expert: int) -> int:
        if self.owner(expert) == rank:
            return expert - rank * self.home_experts
        return self.home_experts + self.slot_expert[rank].index(expert)

    def physical_slot(self, rank: int, expert: int) -> int:
        return rank * self.slots_per_rank + self.local_slot(rank, expert)

    def copy_list(self, rank: int, prev_slot_expert: Optional[Sequence[int]]) -> List[Tuple[int, int]]:
        """(helper slot, expert) pairs rank ``rank`` must fetch from the owner."""
        out = []
        for s, e in enumerate(self.slot_expert[rank]):
            if e >= 0 and (prev_slot_expert is None or prev_slot_expert[s] != e):
                out.append((s, e))
        return out


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def plan_reference(counts: Sequence[Sequence[int]],
                   num_helper_slots: int,
                   block_m: int,
                   prev_slot_expert: Optional[Sequence[Sequence[int]]] = None,
                   open_gate_blocks: int = 0,
                   max_rounds: Optional[int] = None) -> Plan:
    """Load-only block-granular replica planning (MegaFlux, App. A.1).

    ``counts[r][e]`` is the number of tokens rank ``r`` routes to expert ``e``.
    ``prev_slot_expert[r][s]`` is the expert a helper slot currently holds;
    reusing it costs no weight copy, so such ranks are preferred receivers
    and are exempt from ``open_gate_blocks``, the minimum allocation that
    justifies copying weights into a fresh slot.
    """
    R = len(counts)
    E = len(counts[0])
    S = num_helper_slots
    assert E % R == 0 and block_m > 0 and S >= 0
    H = E // R
    owner = [e // H for e in range(E)]
    if prev_slot_expert is None:
        prev_slot_expert = [[-1] * S for _ in range(R)]
    cached = [set(x for x in prev_slot_expert[r] if x >= 0) for r in range(R)]

    total = [sum(counts[r][e] for r in range(R)) for e in range(E)]
    b = [_ceil_div(total[e], block_m) for e in range(E)]
    q = [[0] * R for _ in range(E)]
    for e in range(E):
        q[e][owner[e]] = b[e]
    load = [sum(q[e][r] for e in range(E)) for r in range(R)]
    tau = _ceil_div(sum(b), R)

    def replicas_on(r: int) -> int:
        return sum(1 for e in range(E) if owner[e] != r and q[e][r] > 0)

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
    # current maximum load; prefer the owner, then the least loaded copy.
    max_load = max(load)
    replicas = sorted(((q[e][r], e, r) for e in range(E) for r in range(R)
                       if owner[e] != r and q[e][r] > 0))
    for v, e, r in replicas:
        if q[e][r] != v or v == 0:
            continue
        targets = [owner[e]] + sorted((t for t in range(R)
                                       if t != r and t != owner[e] and q[e][t] > 0),
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
        needed = [e for e in range(E) if owner[e] != r and q[e][r] > 0]
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
        order = [owner[e]] + [r for r in range(R) if r != owner[e] and q[e][r] > 0]
        cum, acc = [], 0
        for r in order:
            acc += q[e][r]
            cum.append((r, acc))
        copies.append(cum)

    return Plan(R, E, S, block_m, q, slot_expert, copies, load)


def remap_reference(plan: Plan, counts: Sequence[Sequence[int]],
                    topk_idx_per_rank: Sequence[Sequence[Sequence[int]]]) -> List[List[List[int]]]:
    """Map every rank's logical ``topk_idx`` to physical slot ids.

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
