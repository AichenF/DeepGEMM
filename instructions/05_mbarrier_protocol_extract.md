### mbarrier instruction

Instruction: PTX `mbarrier` for Warp-Specialized (and Persistent) TMA + WGMMA GEMM on Hopper

---

#### 1. Motivation: lower overhead than CUDA barriers in a queue

The CUDA barrier API is clean, but in a warp-specialized queue it can become expensive:

- You may do many barrier ops per slot per K-tile.
- Arrival-token protocols can force bookkeeping that you don’t strictly need.

`mbarrier` is a lower-level primitive that enables an efficient protocol:

- Track slot reuse via a **software parity bit** \(p \in \{0,1\}\).
- Track TMA completion via **tx-count in bytes** using `complete_tx::bytes`.
- Arrange the protocol so **only one thread per warpgroup** touches `empty[]`, and consumers do not `arrive()` on `full[]`.

---

#### 2. Concept: phases, parity, and tx-count

An `mbarrier` phase completes when:

- arrival count reaches 0, and
- tx-count reaches 0

We reuse each slot across many phases. The **parity bit** tells you which phase you’re waiting for.

---

#### 3. Queue protocol overview: `full[]` + `empty[]`

We use two `mbarrier` arrays per queue slot:

- `full[q]`: producer-owned barrier
  - expected arrivals per phase: **1** (producer thread)
  - tx-count: total bytes of A+B for that slot
  - consumers only **wait** on it (no arrive)

- `empty[q]`: consumer-owned barrier
  - expected arrivals per phase: **num_consumers** (one arrival per consumer warpgroup)
  - producer waits on it before overwriting the slot

Parity handling (typical ring-buffer reuse):

```cpp
int p = 0;
int qidx = 0;
// each time qidx wraps, flip parity (slot reuse enters next phase)
if (++qidx == QSIZE) { qidx = 0; p ^= 1; }
```

---

```cpp
// One-time init (single thread):
if (threadIdx.x == 0) {
  for (int i = 0; i < QSIZE; ++i) {
    init_barrier(&full[i],  1);             // producer-only
    init_barrier(&empty[i], num_consumers); // one arrive per consumer warpgroup
  }
}
__syncthreads();

// Producer (one thread, per slot):
wait_parity(&empty[qidx], p);
expect_bytes(&full[qidx], bytes_total);
tma_load_2d_mbarrier(dstA, tmaA, &full[qidx], colA, rowA);
tma_load_2d_mbarrier(dstB, tmaB, &full[qidx], colB, rowB);

// Consumer (all threads participate in compute, but only tid==0 arrives on empty):
wait_parity(&full[qidx], p);
// TODO: WGMMA compute reading from shared slot qidx
if (tid == 0) arrive(&empty[qidx], 1);
```

---

#### 7. Critical tips / common failure modes

- [ ] `full[]` and `empty[]` are `.b64` aligned to 8 bytes in shared memory.
- [ ] Barriers are initialized exactly once per block before use.
- [ ] Parity bit `p` flips consistently on slot reuse (commonly on `qidx` wrap).
- [ ] `full[]` expected arrivals is **1** (producer only); consumers do not call `arrive()` on it.
- [ ] `empty[]` expected arrivals equals `num_consumers` and **exactly one thread per consumer warpgroup** calls `arrive()` per slot.
- [ ] Producer calls `expect_bytes(full[q], bytes_total)` **before** issuing all copies linked to `full[q]`.
- [ ] The sum of copy bytes linked to `full[q]` equals `bytes_total`.
- [ ] Do not mix CUDA barrier objects with `mbarrier` in the same queue protocol.
- [ ] Do not reinitialize `mbarrier` objects per tile or per K-iteration; reuse phases via parity.
- [ ] If results “sometimes pass, sometimes fail”, suspect: wrong byte count, wrong parity flip, or tensor-map addressability (tma pointer not in device memory).
- [ ] Store mapping uses warpgroup-local indexing (`tid = threadIdx.x % 128`) when writing WGMMA fragments.
