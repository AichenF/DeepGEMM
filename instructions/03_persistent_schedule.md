#### 2. Concept: one worker per SM + a software scheduler

Instead of launching one block per output tile, a persistent GEMM kernel:

- Launches **~one worker CTA per SM** (or per “worker slot”).
- Each CTA computes **multiple** output tiles \((m,n)\) selected by a **software scheduler**.

This is most natural if you already have a warp-specialized (producer/consumer) kernel for a single tile.

---

#### 3. Persistent grid shape (how many workers?)

Instead of launching one block per output tile:

```text
grid.x = (M/BM) * (N/BN)
```

you launch a small grid:

```text
grid.x = NUM_WORKERS   // typically ~SM count (or a grouping-friendly value like 128)
```

and each worker iterates over multiple tiles.

Rule of thumb for `NUM_WORKERS`:

- `NUM_WORKERS = min(total_tiles, sm_count)` is a good default.
- Some schedules prefer a grouping-friendly value (e.g. 128 on a 132-SM GPU).

---

#### 4. Scheduler interface (what you need for correctness)

You need a function that maps a worker index to a sequence of tiles:

```cpp
struct Schedule {
  __device__ Schedule(int M, int N, int worker_idx, int num_workers);
  // returns false when no work remains
  __device__ bool next(int& block_m, int& block_n);
};
```

Minimal contiguous scheduler (simple, always correct with divisibility assumptions):

```cpp
// tile_id in [0, total_tiles)
int tile_id = it * num_workers + worker_idx;
block_m = tile_id / tiles_n;
block_n = tile_id % tiles_n;
```

If you later want L2-locality grouping, keep the same interface and swap the policy.

Correctness properties your scheduler must satisfy:

- **Exactly-once coverage**: every output tile is produced exactly once (no duplicates, no missing tiles).
- **Deterministic iteration**: if both producer and consumers call `schedule.next(...)`, they must see the **same tile sequence** (or you must centralize scheduling so they stay consistent).
- **Termination**: all warpgroups must agree on when work is finished (avoid producer waiting forever on `empty[]` after consumers stop, or consumers waiting forever on `full[]` after producer stops).
