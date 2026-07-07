# SM90 W4A8 grouped GEMM design

## Goal

Add a correctness-first SM90 grouped GEMM path for the
`cuda-practice/grouped_gemm` contract. Tokens are contiguous by expert and the
operation is

```text
D[offsets[e]:offsets[e + 1], :]
    = dequant(A[offsets[e]:offsets[e + 1], :])
      @ dequant(W[e]).T
```

The public inputs are:

- `A`: E4M3 FP8, shape `(M, K)`, with one FP32 scale per token.
- `W`: canonical packed E2M1 FP4, shape `(E, N, K / 2)`.
- `W_scale`: E4M3 FP8, shape `(E, N, K / 16)`.
- `W_global_scale`: FP32, shape `(E,)`.
- `offsets`: INT32 cumulative token counts, shape `(E + 1,)`.
- `D`: BF16, shape `(M, N)`.

All six task shapes must pass the task's elementwise check (`rtol=0.05`,
`atol=2.0`) on H20 (SM90). Performance tuning is deferred until correctness is
established.

## Considered approaches

1. Dequantize weights to BF16 or FP8 during the untimed preparation step and
   call an existing grouped kernel. This is the shortest correctness path, but
   it does not exercise packed NVFP4 at runtime and was explicitly rejected.
2. Add a dedicated SM90 packed-NVFP4 grouped kernel next to the existing SM90
   FP8 grouped kernel. Repack only the byte layout during preparation, load
   packed FP4 and block scales in the main kernel, expand them to FP8 in shared
   memory, and feed Hopper WGMMA. This is the selected design.
3. Port the complete `megamoe_nvfp4` implementation. It contains the desired
   dequantization pattern, but it also carries dispatch/combine, two GEMM
   phases, SwiGLU, symmetric buffers, and distributed scheduling that are not
   part of this task. It will be used as a read-only reference instead.

## Architecture

The implementation introduces an independent `m_grouped_nvfp4_gemm_nt_contiguous`
API rather than changing the behavior of existing FP8/FP4 APIs.

The preparation helper transforms canonical `(E, N, K / 2)` packed weights and
`(E, N, K / 16)` scales into a tile-major representation suitable for
coalesced loads. It may fuse packed values and their scale bytes into one
storage tile, but it must not dequantize values. The per-expert FP32 global
scale remains a separate tensor.

The CUDA kernel uses a fixed correctness-first tile and one Hopper WGMMA math
warpgroup. For each K tile it:

1. loads activation FP8 bytes and packed weight/scale bytes;
2. expands E2M1 values multiplied by their per-16 E4M3 scales into a two-term
   FP8 representation in shared memory;
3. executes two FP8 x FP8 WGMMA passes with FP32 accumulation;
4. multiplies each output row by its token activation scale and the compensated
   expert global weight scale;
5. stores only rows inside the expert's true, unpadded token range as BF16.

Applying the token and global scales in the epilogue is algebraically valid
because each is constant across the reduction dimension for its row/expert.
The two-term representation is required because the task normalizes block
scales into nearly the full E4M3 range: `E2M1 * block_scale` can reach
`6 * 448`, which would saturate a single FP8 operand. The kernel represents
`E2M1 * block_scale / 8` as `fp8_primary + fp8_residual` and multiplies the
global scale by eight in the epilogue. The power-of-two normalization is exact
and the residual WGMMA pass provides enough precision for the elementwise
acceptance criterion.

## Group scheduling and tails

`offsets` are not host-synchronized. A persistent device scheduler scans at
most 32 experts to map a linear output tile to `(expert, local_m_tile,
n_tile)`. The number of valid M tiles for expert `e` is
`ceil((offsets[e + 1] - offsets[e]) / BLOCK_M)`. Empty experts contribute no
work.

The last tile of an expert may read activation rows belonging to the next
expert, but those lanes are never stored. The final expert relies on normal
global bounds checks. N and K are aligned for every task shape; explicit shape
checks reject unsupported layouts rather than silently producing wrong data.

## Validation

Validation is layered:

1. Exhaustively compare the device E2M1/E4M3 dequantization primitive for all
   16 FP4 codes and representative/all E4M3 scale codes.
2. Compare one-expert and multi-expert small cases with a PyTorch reference,
   including empty experts and non-multiple-of-`BLOCK_M` tails.
3. Reproduce all six `cuda-practice/grouped_gemm` shapes and use its exact
   elementwise tolerance and NaN/Inf checks.
4. Run existing DeepGEMM tests affected by API registration and JIT compilation.

The first implementation is complete only when all six authoritative task
shapes pass on H20.
