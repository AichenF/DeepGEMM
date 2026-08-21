# SM120 DeepSeek V4 Flash MegaMoE G8

This directory contains a standalone, single-entry SM120 MegaMoE implementation
for the routed-expert shape used by DeepSeek V4 Flash. It is a qualification
artifact rather than a DeepGEMM JIT/API integration.

## Scope

- 384 routed experts across 8 ranks, 48 local experts per rank
- top-k 6 routing
- hidden size 7168, intermediate size 3072, output size 7168
- FP8 activations, FP4 weights, UE8M0 block scales, BF16 output
- three epochs using result slots `[0, 1, 0]`
- no shared-expert path

The kernel launches 110 cooperative CTAs with 384 threads each. CTA 0 owns
NCCL GIN service, result return, fixed-slot combine, and acknowledgement.
CTAs 1-109 execute the canonical SM120 DeepGEMM math path. Work is scheduled in
G8 chunks: six W1 chunks and seven W2 chunks per M64 task, with eight physical
N128 tiles in each chunk. The single global entry covers dispatch, task build,
W1, SwiGLU/requantization, W2, remote return, and combine.

The transport protocol captures signal baselines before worker/service
divergence, uses strong-signal puts, and proves two-slot reuse with a
post-combine acknowledgement. It does not issue a flush in the steady-state
protocol.

## Files

- `cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu`: kernel entry
  and device protocol
- `deepgemm_fp8_fp4_mega_moe_sm120_production_host.cu`: shared fixtures,
  registered buffers, GIN setup, and independent references
- `deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_host.cu`:
  fail-closed correctness executable
- `deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_perf_host.cu`:
  event-timed executable with pre/post correctness audits
- `run_sm120_canonical_fused_ready_chunk8_perf.py`: distributed benchmark and
  result validator
- `selected-matrix-receipt.json`: correctness qualification summary
- `qualification-manifest.json`: immutable source, build, and evidence hashes

The SM120 math implementation is provided by
`deep_gemm/include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh`; its qualified
hash is recorded in the manifest.

## Build

The qualified build used CUDA 13.3, NCCL 2.30.7 with GDAKI, and `sm_120a`.
With `NCCL_ROOT` pointing to the NCCL installation:

```bash
ROOT=$(git rev-parse --show-toplevel)
Q=$ROOT/examples/sm120_megamoe/qualified_g8
COMMON=(
  -std=c++20 -O3 --expt-relaxed-constexpr
  --generate-code=arch=compute_120a,code=sm_120a
  -lineinfo -Xptxas=-v,-warn-spills
  -I"$Q"
  -I"$ROOT/deep_gemm/include"
  -I"$ROOT/third-party/cutlass/include"
  -I"$ROOT/third-party/cutlass/tools/util/include"
  -I"$NCCL_ROOT/include"
)
LIBS=(
  -L"$NCCL_ROOT/lib" -Wl,-rpath,"$NCCL_ROOT/lib"
  -lnccl -lcuda -lcudart -lcudadevrt -Xcompiler=-pthread
)

nvcc "${COMMON[@]}" \
  "$Q/deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_host.cu" \
  "${LIBS[@]}" \
  -o build/sm120_megamoe_g8_correctness

nvcc "${COMMON[@]}" \
  "$Q/deepgemm_fp8_fp4_mega_moe_sm120_production_canonical_fused_ready_chunk8_perf_host.cu" \
  "${LIBS[@]}" \
  -o build/sm120_megamoe_g8_perf
```

## Correctness gate

A minimal task-bearing test is:

```bash
CUDA_VISIBLE_DEVICES=0 \
NTHREADS=1 \
NCCL_GIN_TYPE=3 \
NCCL_NET_PLUGIN=spcx \
CAKE_ACTIVE_ROWS=1 \
CAKE_MASK_PERIOD=0 \
CAKE_ROUTE_MODE=balanced \
CAKE_ORACLE=distinct_k32 \
build/sm120_megamoe_g8_correctness
```

Success requires process exit 0, `status=pass`, `exact_bf16_equal=true`, zero
stage/protocol/signal/ack/output/guard mismatches, and one kernel launch per
epoch. The selected matrix covers 9 cases and 31 rank records through world
size 8 and 2048 active rows. Seven cases were rerun from the cleaned source;
the two P8 cases are inherited from the original qualification because the
complete CUDA SASS dumps are byte-identical. The exact boundary and hashes are
recorded in `selected-matrix-receipt.json`.

## Performance gate

After the selected-matrix receipt passes:

```bash
python "$Q/run_sm120_canonical_fused_ready_chunk8_perf.py" \
  --binary build/sm120_megamoe_g8_perf \
  --matrix-receipt "$Q/selected-matrix-receipt.json" \
  --world-size 8 \
  --active-rows 2048 \
  --oracle distinct_k32 \
  --route-mode balanced \
  --mask-period 0 \
  --warmup 5 \
  --repeat 100 \
  --output build/sm120_megamoe_g8_r100.json
```

The qualified P8/R2048 run measured 23.393127 ms mean, 23.358768 ms P50,
24.978671 ms P95, and 555.205 aggregate tensor TFLOP/s. The event contains one
full-chain kernel per sample; communicator rendezvous and correctness audits
are outside the event. A final same-node paired check on four otherwise idle
GPUs measured 3.822493 ms for the reference and 3.814461 ms for this baseline,
a `-0.210%` delta within normal run-to-run variation. Their complete CUDA SASS
dumps are byte-identical.

## Qualification limits

The selected matrix is exact-BF16 clean, but the executable intentionally keeps
its formal `functional_qualified`, `resource_qualified`, and
`performance_qualified` flags false. The register-repartition instruction was
not retained by the compiler, so this is not a production resource or
performance sign-off. The qualified resource record is 128 registers, 8 bytes
of stack, zero spills, zero local memory, one barrier, and 94,208 bytes of
dynamic shared memory. Shared experts and public DeepGEMM API/JIT integration
are deferred.
