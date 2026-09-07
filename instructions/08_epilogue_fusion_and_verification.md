#### 0. Motivation: why epilogue fusion, and why a dedicated epilogue warpgroup?

Epilogue fusion means: after WGMMA finishes producing fp32 accumulators `d[...]`, apply an operation (pointwise or reduction) in fp32, then write the final output.

Compared to a non-fused kernel that stores `d[...]` directly to global memory, a fused epilogue commonly needs:

- **More work per output element** (activation, conversion, layout handling, reductions)
- **A clean separation** between “math” and “store/fuse” so the WGMMA warpgroup stays simple

That’s why many high-performance Hopper kernels use **producer + math + epilogue** warpgroups:

- Math warpgroup stays “pure WGMMA” and writes a shared fp32 mailbox
- Epilogue warpgroup reads the mailbox, performs fusion, and stores the final result

#### 4. Fusion verification checklist

- [ ] Uses the fusion style required by the run (warp-specialized epilogue vs naive post-WGMMA).
- [ ] Fusion runs in fp32 (before bf16/fp16 conversion).
- [ ] For `m64n64`: handles both `row` and `row+8` fragments.
- [ ] Warp-specialized epilogue: the handoff barrier (named barrier, `mbarrier`, etc.) prevents mailbox races.
- [ ] No `__syncthreads()` inside divergent `if (wg_idx==...)` branches.
- [ ] `mean` with `BN < N`: uses cross-CTA accumulation (e.g. `atomicAdd(partial_sum / N)` into `[M,1]` fp32 output).
- [ ] If using a TMA queue (`full[]/empty[]`): barrier participant counts match exactly the warpgroups that arrive in the loop (do not include epilogue WG unless it truly participates every iteration).

- [ ] Descriptor leading dimensions match actual matrix strides
- [ ] Descriptor matrix size parameters are correct for tile dimensions
- [ ] TMA tensor map shape is [width, height] = [minor_dim, major_dim]
- [ ] TMA coordinates are (col_offset, row_offset)
- [ ] Shared memory indexing advances correctly in K dimension
- [ ] Both descriptor and TMA use 128B swizzle
- [ ] All threads participate in barrier synchronization
- [ ] Output store pattern matches WGMMA register distribution
- [ ] TransA/TransB flags match actual matrix layouts in shared memory
