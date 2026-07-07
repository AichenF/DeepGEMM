#### 1. Motivation: overlap TMA loads with tensor-core compute

TMA loads (global → shared) have latency. If a single warpgroup does:

- load A/B tile
- wait for completion
- run WGMMA

then WGMMA execution is often blocked by loads.

Warp specialization turns the kernel into a producer/consumer pipeline:

- **Producer warpgroup** keeps loading upcoming tiles into a shared-memory ring buffer.
- **Consumer warpgroup(s)** keep running WGMMA as soon as tiles are ready.

With `QSIZE > 1`, producer and consumers can run mostly independently, hiding load latency.

#### 3. Required queue protocol (per-slot)

Producer (per K-tile, per slot):

1. Wait until slot is empty: `empty[q].wait(empty[q].arrive())`
2. Issue TMA loads for A/B into that slot, associating them with `full[q]`
3. Mark the total bytes via `barrier_arrive_tx(full[q], 1, bytes_total)`

Consumer (per K-tile, per slot):

1. Wait until slot is full: `full[q].wait(full[q].arrive())`
2. Run WGMMA using `sA[q]`, `sB[q]`
3. Release slot: `empty[q].arrive()`

#### 4. Practical notes (layout/indexing + tuning)

These are common sources of subtle correctness/perf issues:

- **Indexing consistency**: Your tensor-map encoding (global shape/stride), your shared-memory indexing, and your WGMMA `TransA/TransB` flags must all agree.
- **Queue depth** (`QSIZE`): typical values are **2–8**. Larger `QSIZE` improves latency hiding but increases dynamic shared memory.
- **Single-thread producer**: start with `wg_idx==0 && tid==0` issuing TMA to avoid duplicated loads and complicated barrier accounting.
- **Dynamic shared memory**: ring buffers typically require dynamic shared memory; make sure host launch sets both:
  - the dynamic shared memory **bytes**, and
  - `cudaFuncAttributeMaxDynamicSharedMemorySize` for the kernel.

