# SM120 MegaMoE — M64xN32 two-stage residency schedule, EP-generic release (EP4 and EP8)

One generated MoE megakernel per variant, valid for 4-rank and 8-rank expert parallelism. Each build entry point
exports a unique symbol and enforces its exact EP width. The two EP widths come from the same generator source, which
resolves the expert-parallel width at generation time (experts per rank, peer count and every derived buffer size
become literals), so one file is emitted per EP width. Both variants keep all EP-specific tuning inside the generator:
the EP4 emission is the EP4-tuned kernel, the EP8 emission is the EP8-tuned kernel, and no EP4 mechanism leaks into
the EP8 build (see "How the EP policy was decided").

| file | variant | EP | bytes | lines | ptxas (sm_120a) | sha256 |
|---|---|---|---:|---:|---|---|
| `cake_sm120_megamoe_m64n32_stage2_residency_fp8shared_ep4.cu` | fp8shared | 4 | 298143 | 5063 | 167 REG, 0 spills, 16 barriers, 8 B stack | `2c5f463af0e1efbc8f251c46c75d38c6eb8a1a588dc5e530c6ca771daca34e76` |
| `cake_sm120_megamoe_m64n32_stage2_residency_fp8shared_ep8.cu` | fp8shared | 8 | 294559 | 5019 | 167 REG, 0 spills, 16 barriers, 8 B stack | `a9eae1b4fe00e345cfd96296b75585e4149de4eca3f243249529d93eed7ed764` |
| `cake_sm120_megamoe_m64n32_stage2_residency_noshared_ep4.cu` | noshared | 4 | 252 | 4 | 167 REG, 0 spills, 16 barriers, 8 B stack | `472088f2d0496f38b6b8bc1ccfbf7b4e660fec5224bdeec3c86b46124994bfb2` |
| `cake_sm120_megamoe_m64n32_stage2_residency_noshared_ep4_r2048_unroll2.cu` | noshared, unroll-2 instance | 4 | 266 | 4 | 168 REG, 0 spills, 16 barriers, 8 B stack | `7d4a78c64ed400aba865c173e81a7dda224999e3e716d8dbdd4a34eb9457cdef` |
| `cake_sm120_megamoe_m64n32_stage2_residency_noshared_ep8.cu` | noshared | 8 | 214094 | 3882 | 164 REG, 0 spills, 16 barriers, 8 B stack | `33a3ed7c55639eaaa770f0be3cb1f6f38ccb6950a54d268e50e1d5a8b5251680` |

Resource figures are from `nvcc -cubin --generate-code=arch=compute_120a,code=sm_120a -std=c++17 -O3 --resource-usage`
on each build entry point as shipped. The two EP4 entry points include the 260232-byte, 4502-line common body
`cake_sm120_megamoe_m64n32_stage2_residency_noshared_ep4.cuh` (SHA256
`65f4bb47925738944939756e3b7e35930a6936d2754f1b7818af8b78f95751bc`) and select K-loop unroll 1 or 2 at compile
time. This removes a 4493-line duplicate without changing either kernel's SASS. The host selects unroll 2 when the
per-rank row count is 2048 and unroll 1 otherwise. At EP8 a single instance is used.

## Geometry (both variants, both EP widths)

DeepSeek-V4-Flash MoE layer: hidden 4096, intermediate 2048, 256 routed experts, top-6, one shared expert.
EP8: 32 local experts per rank, 8 ranks; EP4: 64 local experts per rank, 4 ranks. Routed experts run W4A8
(MXFP8 E4M3 activations x MXFP4 E2M1 weights, UE8M0 K32 scales, FP32 accumulation), SwiGLU with the released
clamp, BF16 output, lossless 12-bit result codec on the return path. Transport: NCCL GIN device API over the RDMA
NICs (one persistent cooperative kernel per layer: push dispatch of 4352 B records in 8 chunked signalled puts per
peer, unified W1/W2 tile stream, chunked signalled return puts, in-kernel combine).

- `fp8shared` computes the shared expert in-kernel on the checkpoint's FP8 E4M3 weights (exact weights, MXFP8
  activations) and returns the full MoE output (routed top-6 sum + shared expert).
- `noshared` returns the routed top-6 sum only; the caller adds the shared expert.

## Correctness

Every emission validates bit-exactly (3 epochs, all ranks) against the harness oracle on real layer-20 weights with
(a) captured real activations and real gate routing and (b) Gaussian activations, at EP4 on 4 GPUs and at EP8 on
8 GPUs. The two EP emissions of one variant are the same schedule; they differ only in the literals derived from
the EP width (experts per rank 64 vs 32, peer count 4 vs 8, signal bases and window sizes).

## Measured performance

Platform: 8x sm_120a GPUs (110 SMs each, no NVLink), PCIe Gen5, one ConnectX-8 400G RDMA port per GPU, NCCL 2.30.7
GIN. Protocol: L2 flushed before every timed iteration (the serving-realistic state; warm numbers are within 2 %),
64 warm-up + 21 timed iterations, CUPTI first-kernel-start to last-kernel-end, per-iteration maximum across ranks
then median. "real" = real layer-20 weights + captured text activations + real gate; "Gaussian" = same weights,
N(0,1) activations. Rows = tokens per rank. Milliseconds per MoE layer.

### EP4 (4 GPUs), the kernels in this directory

| variant | fixture | R512 | R1024 | R2048 | R4096 | R8192 |
|---|---|---:|---:|---:|---:|---:|
| noshared | real | 1.20 | 1.68 | 2.60 | 4.88 | 9.69 |
| noshared | Gaussian | 1.21 | 1.56 | 2.38 | 4.19 | 8.78 |
| fp8shared (incl. shared expert) | real | 1.25 | 1.76 | 2.83 | 5.72 | 10.60 |
| fp8shared (incl. shared expert) | Gaussian | 1.26 | 1.66 | 2.59 | 4.73 | 9.42 |

Same platform, same fixture, same protocol, routed-only paths: the DeepEP-v2 + DeepGEMM production MoE path (W4A8
arm, CUDA-event dispatch-start to combine-end, shared expert outside the timed boundary) measured 3.08 / 5.09 /
9.46 / 17.83 ms at R1024-R8192 (1.8-2.0x the noshared kernel above), the FlashInfer SM120 CuTeDSL split megakernel
2.62 / 4.71 / 9.30 / 17.87 ms.

### EP8 (8 GPUs), the kernels in this directory

| variant | fixture | R512 | R1024 | R2048 | R4096 | R8192 |
|---|---|---:|---:|---:|---:|---:|
| noshared | real | 1.26 | 2.05 | 3.48 | 6.66 | 13.51 |
| noshared | Gaussian | 0.98 | 1.53 | 2.86 | 6.01 | 11.47 |
| fp8shared (incl. shared expert) | real | 1.23 | 2.02 | 3.52 | 6.85 | 13.85 |
| fp8shared (incl. shared expert) | Gaussian | 1.03 | 1.61 | 2.85 | 5.73 | 11.71 |

The noshared EP8 file is byte-identical to the EP8-tuned specialist kernel these numbers were taken on. The
fp8shared EP8 file is the EP-generic emission; measured against the EP8-tuned specialist in alternating pairs it is
0.995-1.003x on real text (5 pairs), 0.963-1.006x on a second real corpus (GSM8K activations, 3 pairs) and
0.987-0.996x on Gaussian (3 pairs), i.e. within noise or slightly faster.

## How the EP policy was decided

The EP4-tuned kernels were derived from the EP-generic kernel by an optimisation campaign that added six
mechanisms (fp8shared: 64-row A/scale TMA copies for short tasks, batched epilogue completion; noshared: the same
64-row copies, epilogue offload to the scatter warps, a K-loop unroll-2 instance for R2048, a wider 128-route return
band). Each mechanism sits behind a generation-time switch that defaults to "EP4 only". At EP8 every mechanism was
ablated individually on real text (paired, 2-3 repetitions): none gave a consistent gain, and two cost 1-5 % on
some shapes (the 64-row copies on every shape, the epilogue offload at R4096), so all six stay off at EP8. The EP8
emission was then specialised so that it is byte-identical to the EP8-tuned kernel; the EP4 emission is
byte-identical to the EP4-tuned kernels. Decisions used real activations as the primary evidence; Gaussian data
served only as a regression guard.

## Not included here — host integration notes

This directory carries the device translation units and this report only. The launch ABI is the M64xN32 residency
ABI of the previous release of this schedule: `fp8shared` takes two extra tensor-map arguments for the shared
expert's W1 (gate|up, 8-interleaved physical row order) and W2 weights whose UE8M0 K32 scales live in segment 32 of
the W1/W2 scale tensors; `noshared` omits the `shared_out` argument and returns the routed top-6 sum in
`final_output`. The world size is a runtime argument but must match the EP width the file was generated for
(the kernel checks it and refuses otherwise); weight and scale tensors carry 64 (EP4) or 32 (EP8) local-expert
segments. The result-slot layout, per-(source, chunk) dispatch and return signal allocation, and launch-time grid
resolution follow the previous README of this schedule. The integrated `sm120_fp8_fp4_routed_moe` path in this
repository corresponds to the EP8 noshared emission; extending it to EP4 requires the EP4 emission plus the EP-width
literals in its layout header (experts per rank, world size, window sizes).
