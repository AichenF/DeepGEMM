# Dense-only NVFP4 small-M warp-MMA design

## Goal

Remove the active complementary 2:4 sparse implementation completely, then
raise the bound-aware SOL of the official memory-bound
`128x4096x7168/E8` shape to at least 70% without changing persistent packed
NVFP4 storage.  On the H20-3e used by Cupra, the accepted roofline constants
are 296 TFLOP/s FP8 and 4.814304 TB/s HBM bandwidth.  The shape's algorithmic
memory lower bound is 27.85 us, so the latency gate is 39.79 us.

The existing `8192x4096x7168/E8` dense dual-M path must remain at or above
80% compute SOL (at most 2.032 ms).

## Why the current small-M path cannot meet the target

SM90 WGMMA has a fixed M64 dimension.  The 128 tokens are distributed across
eight experts, with roughly 0--42 tokens per expert in the benchmark inputs.
Rounding every non-empty expert to M64 executes 448--512 physical rows for
only 128 useful rows.  The resulting 89--102 us tensor-core lower bound is
already above the 39.79 us target, before packed-weight loads, NVFP4 decode,
or synchronization.  Tuning WGMMA tile N or pipeline depth cannot remove this
lower bound.

## Production sparse removal

Delete the active sparse kernel and its dedicated dequantization tables:

- `deep_gemm/include/deep_gemm/impls/sm90_nvfp4_grouped_gemm_sparse.cuh`
- `deep_gemm/include/deep_gemm/quantization/nvfp4_sparse_dequant.cuh`

Remove `SparseFP8MMASelector` from the SM90 MMA helper.  Simplify the JIT
launcher by removing `use_sparse`, `sparse_consumers`, sparse code generation,
sparse shared-memory sizing, and the sparse kernel-name branch.  The public
Python/C++ API and packed-weight layout remain unchanged.  Historical design
documents, experiments, and iteration records remain as audit history.

## Small-M architecture

Use legacy warp-level FP8 `mma.sync.aligned.m16n8k32` with a transposed logical
GEMM orientation:

```
W_tile[16, K] @ A_expert[K, token_tile=8]
    -> C_tile[16 output columns, 8 tokens]
```

The N8 MMA dimension makes token padding eight rather than sixty-four.  Across
the standard benchmark seeds this reduces physical token rows from 448--512
to 144--168.  The physical tensor-core lower bound becomes 28.6--33.3 us,
which leaves a narrow but real path to the 39.79 us latency gate.

Each warp owns one expert and one 16-column output tile.  It iterates over all
8-token groups for that expert.  For each K32 step it reads a packed NVFP4
weight fragment once, applies the two K16 block scales, and reuses the decoded
weight fragment across every active token group.  This avoids multiplying HBM
weight traffic by the number of token tiles.  Zero-token experts are skipped;
partial token groups are zero-filled and stores are predicated.

The bring-up implementation may stage decoded E4M3 fragments in shared memory
to validate fragment mapping.  The target implementation removes that round
trip and forms MMA A-operand registers directly from packed E2M1 nibbles and
the existing E2M1-times-UE4M3 lookup table.  Activations form the MMA B operand.
No decoded FP8 weight is written to global memory.

The existing dense WGMMA kernels remain the fallback for medium and large M.
The initial small-M dispatch is restricted to the official low-token regime so
that the validated large-M dual-consumer path is unchanged.

## Arithmetic

For the dense and warp-MMA paths alike, the transient weight value is

```
B_stage = round_E4M3(E2M1(code) * block_scale / 8)
```

and the epilogue applies

```
output = BF16(accum * activation_scale * global_scale * 8)
```

This intentionally preserves the current single-E4M3 numerical behavior.  The
known K=7168 reference mismatch is reported but is not the optimization gate;
NaN, out-of-range access, expert-routing, and synchronization regressions are
not permitted.

## Alternatives rejected

1. Keeping or merely disabling sparse code leaves dead production machinery
   and contradicts the requested complete removal.
2. Retuning M64 WGMMA cannot cross the physical padding lower bound.
3. Persistent global FP8 expansion would ease MMA input handling but violates
   the packed-NVFP4 storage requirement and increases model weight bytes.
4. A scalar/SIMT dot product avoids padding but cannot supply the required
   useful throughput for the K7168 shape.

## Validation

1. Active `csrc`, `deep_gemm/include`, `tests`, and `examples` contain no
   NVFP4 sparse kernel, selector, LUT, include, or dispatch reference.
2. The extension and fresh JIT kernels compile on the H20-3e.
3. Standard Cupra timing is used: eight fresh input slots, ten warmups, and
   thirty cold-L2 timed iterations.
4. `128x4096x7168/E8` is at most 39.79 us and therefore at least 70% roofline
   SOL under the accepted 296-TFLOP/s and 4.814304-TB/s constants.
5. `8192x4096x7168/E8` remains at most 2.032 ms and at least 80% compute SOL.
6. All six shapes are reported.  The four K2048 shapes must retain structural
   correctness; known K7168 quantization errors remain explicitly reported.
7. Prepared bytes, dtypes, and numel match the packed-FP4 input exactly; no
   persistent FP8 weight copy is introduced.

