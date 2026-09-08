# FlashInfer Cutlass Humming BF16 Serving-Layer Benchmark

## Goal

Measure current FlashInfer-main `CutlassHummingConfig` against the selected
same-source TP4 multi-kernel MXFP4 MoE implementation for the DeepSeek-V4-Flash
rank-local shape. This is a supplemental serving-layer comparison starting
from BF16 activation; it does not change the production FP8-input objective.

## Compared contracts

Both paths receive the same BF16 `X[M,4096]`, precomputed top-k6 IDs and
weights, and logically identical raw OCP MXFP4/E8M0 weights. Raw weights are
generated once per rank and transformed outside timing into the custom tiled
layout and FlashInfer Humming mixed-input layout.

FlashInfer is used without kernel changes. Its route expansion performs
per-expanded-token rowwise BF16-to-E4M3 quantization and then executes the
CUTLASS Humming W13, activation, W2 and finalize path. The custom graph adds
the existing Humming group-128 BF16-to-E4M3 input quantizer before its selected
route-align, W13, SwiGLU/requant, W2 and local weighted reduction. Both local
outputs feed the same SGLang `CustomAllReduceV2` communicator.

The online quantizers have different scaling granularity. Therefore latency
is the primary serving-contract result, while correctness is checked both
against each backend's quantized reference and against a common dequantized
MXFP4 BF16 reference with an explicitly reported error envelope.

## Alternatives considered

1. Native serving paths, selected: retains `CutlassHummingConfig` unchanged and
   measures deployable behavior.
2. Patch FlashInfer to consume group-128 prequantized X: most numerically exact,
   but no longer tests `CutlassHummingConfig` as requested and duplicates the
   existing strict FP8-input Humming comparison.
3. Quantization-free comparison: already covered by the existing original
   Humming benchmark and does not answer this BF16 serving-layer question.

## Measurement protocol

- Shape: E=256, H=4096, global I=2048, TP4 local I=512, top-k=6.
- M: 8, 16, 32, 64 and 128; identical random routes across ranks/backends.
- CUDA Graph for the full timed path, including input quantization and CARv2.
- Separate 256 MiB same-stream L2 eviction before every replay; eviction is
  excluded from CUDA-event timing.
- Paired balanced execution order, at least five outer batches, and max-rank
  latency. Report min/median/max, per-batch medians and geometric mean.
- Exclude weight conversion, allocation, JIT, tactic selection, router/top-k
  computation and graph capture.
- Capture one M=8 and one M=128 replay with Nsight Systems to enumerate all
  timed device kernels and explain any gap.

## Validation and failure policy

First run a single-rank local smoke test, then TP4 correctness for every M.
Reject performance numbers if either backend produces non-finite output, route
metadata differs, CARv2 disagrees with an NCCL all-reduce oracle, or the graph
contains an unintended allocation/copy/reset. A backend build or supported-
shape failure is recorded as evidence rather than silently replaced by a
different FlashInfer backend.

All new source, logs, version manifests and profiler artifacts live under
`/home/xutingz/fac`; existing dirty files and the read-only DeepGEMM checkout
are not modified.
