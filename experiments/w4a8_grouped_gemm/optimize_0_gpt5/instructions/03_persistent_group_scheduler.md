#### 2. Concept: one worker per SM + a software scheduler

Instead of launching one block per output tile, a persistent GEMM kernel:

- Launches **~one worker CTA per SM** (or per “worker slot”).
- Each CTA computes **multiple** output tiles \((m,n)\) selected by a **software scheduler**.

This is most natural if you already have a warp-specialized (producer/consumer) kernel for a single tile.

Rule of thumb for `NUM_WORKERS`:

- `NUM_WORKERS = min(total_tiles, sm_count)` is a good default.
- Some schedules prefer a grouping-friendly value (e.g. 128 on a 132-SM GPU).

Correctness properties your scheduler must satisfy:

- **Exactly-once coverage**: every output tile is produced exactly once (no duplicates, no missing tiles).
- **Deterministic iteration**: if both producer and consumers call `schedule.next(...)`, they must see the **same tile sequence** (or you must centralize scheduling so they stay consistent).
- **Termination**: all warpgroups must agree on when work is finished (avoid producer waiting forever on `empty[]` after consumers stop, or consumers waiting forever on `full[]` after producer stops).

## Summary of Key Insights

1. **Single kernel, multiple problems**: Amortize launch overhead
2. **Pointer arrays**: Flexible memory layout for variable sizes
3. **Persistent loop**: Threadblocks process many tiles via work-stealing
4. **Tile-based work division**: Natural unit for parallelization
5. **Two scheduling modes**: Dynamic (device) vs static (host precompute)
6. **Load balancing via sorting**: Put large-K problems first
7. **Warp-level primitives**: Efficient scheduling with `__shfl_sync` and `__ballot_sync`

