# SM120 MegaMoE shape family

A single-entry SM120 MegaMoE implementation whose routed-expert extents are a
build-time configuration rather than a property of the source. One translation
unit now serves every supported shape and expert-parallel width, replacing the
previous practice of copying the generated kernel per shape.

The default configuration is DeepSeek V4 Flash routed experts:

- hidden 4096, intermediate 4096, output 4096
- 256 global experts, top-k 6
- MXFP8 E4M3 activations, MXFP4 E2M1 weights, UE8M0 K32 scales, W4A8 GEMM
- BF16 output, three epochs over result slots `[0, 1, 0]`
- no shared-expert path

The kernel launches 110 cooperative CTAs of 384 threads. CTA 0 owns NCCL GIN
service, result return, fixed-slot combine and acknowledgement; CTAs 1-109 run
the canonical SM120 DeepGEMM math path. The single global entry covers dispatch,
task build, W1, SwiGLU/requantization, W2, remote return and combine.

## Configuration

`megamoe_shape_config.cuh` owns every extent. Override a knob with `-D` and all
window strides, task bounds, tile counts, pool extents and protocol offsets
follow:

| Macro | Default | Meaning |
| --- | --- | --- |
| `CAKE_MOE_HIDDEN` | 4096 | hidden size, also the dispatch activation row |
| `CAKE_MOE_INTERMEDIATE` | 4096 | per-expert intermediate size |
| `CAKE_MOE_OUTPUT` | `CAKE_MOE_HIDDEN` | output size |
| `CAKE_MOE_TOPK` | 6 | routes per token |
| `CAKE_MOE_LOCAL_EXPERTS` | 32 | experts owned by one rank |
| `CAKE_MOE_PHYSICAL_RANKS` | 8 | rank capacity of the registered windows |
| `CAKE_MOE_MAX_ROWS` | 2048 | tokens per rank |
| `CAKE_MOE_RING_SLOTS` | 2 | result/dispatch slots proving reuse |
| `CAKE_MOE_COMBINE_CTAS` | 110 | cooperative grid width |

Global experts are `world_size * CAKE_MOE_LOCAL_EXPERTS`, so holding 256 experts
while narrowing expert parallelism means widening the local expert count:

```bash
./build.sh out/ep8                                  # EP8, 8 x 32 = 256 experts
./build.sh out/ep4 -DCAKE_MOE_LOCAL_EXPERTS=64      # EP4, 4 x 64 = 256 experts
./build.sh out/g8  -DCAKE_MOE_HIDDEN=7168 \
                   -DCAKE_MOE_INTERMEDIATE=3072 \
                   -DCAKE_MOE_LOCAL_EXPERTS=48      # the earlier G8 geometry
```

`build.sh` compiles both executables inside the pinned CUDA 13.3 container for
`sm_120a` and records ptxas logs, `cuobjdump` resource usage, SASS and a hash
manifest next to them.

## Files

- `megamoe_shape_config.cuh`: the single source of shape truth
- `cake_sm120_megamoe_production_canonical_fused_ready_chunk8.cu`: kernel entry,
  math donor instantiation and device protocol
- `deepgemm_fp8_fp4_mega_moe_sm120_production_host.cu`: shared fixtures,
  registered buffers, GIN setup and independent references
- `..._chunk8_host.cu`: fail-closed correctness executable
- `..._chunk8_perf_host.cu`: timed executable with pre/post correctness audits
- `build.sh`: containerized build for one configuration
- `run_correctness_matrix.py`: fail-closed correctness gate and receipt writer
- `run_perf_abba.py`: paired CUPTI max-rank performance gate

## Correctness gate

```bash
./build.sh out/ep4 -DCAKE_MOE_LOCAL_EXPERTS=64
python3 run_correctness_matrix.py \
  --build out/ep4 --gpus 0,1,2,3 --world-size 4 \
  --hidden 4096 --intermediate 4096 --experts 256 --topk 6 \
  --output evidence/ep4-correctness-matrix.json
```

The matrix covers balanced, skewed and empty routing; 1, 17, 113, 1024 and 2048
active rows; masked-route periods 3 and 7; and the zero, analytic and
`distinct_k32` oracles. A case passes only on exit status zero, one kernel launch
per epoch, bit-exact BF16 output, a stable per-epoch route total, the `[0, 1, 0]`
slot-reuse pattern and zero protocol, owner, counter, signal, acknowledgement,
ready-scheduler, stage, output and guard mismatches.

## Performance gate

```bash
docker run --rm -v $PWD:/src:ro -v $PWD/out:/out nvcr.io/nvidia/pytorch:26.07-py3 \
  nvcc -O2 -std=c++17 -Xcompiler=-fPIC -shared /src/cupti_kernel_trace.cpp \
  -I/usr/local/cuda-13.3/targets/x86_64-linux/include \
  -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcupti \
  -o /out/libcake_cupti_trace.so

python3 run_perf_abba.py \
  --arm baseline=out/ep4 --arm candidate=out/ep4-candidate \
  --injection out/libcake_cupti_trace.so \
  --gpus 0,1,2,3 --world-size 4 --rows 2048 \
  --hidden 4096 --intermediate 4096 --experts 256 --topk 6 --repeat 20 \
  --output evidence/ep4-abba.json
```

`cupti_kernel_trace.cpp` is a CUPTI injection library, so the timed executable
carries no instrumentation. Latency is the CUPTI concurrent-kernel activity
envelope of one iteration on each rank; a sample is reduced with a max across
ranks before any statistic, and the two arms run in A/B/B/A order. Because the
envelope is reconstructed from activity records rather than read from a CUDA
event around one launch, a multi-kernel candidate is measured on the same basis
as the fused kernel, with launch gaps and inter-kernel idle time included.
For a partially fused implementation, pass one ordered `--arm-kernel
candidate:KERNEL_SUBSTRING` argument per dependent kernel. The runner rejects a
trace with a missing, duplicated or reordered kernel instead of silently timing
only part of an iteration. Both runners also reject a build whose recorded
compile-time shape or expert partition differs from the requested workload.

A null A/A run of the EP4 build resolves a 0.18% median difference with 0.06%
endpoint drift, which sets the floor for a believable improvement.

## Equivalence to the previous per-shape sources

At the default configuration this source compiles to SASS that is identical to
the sealed 4096/4096/E256 artifact, modulo the anonymous-namespace hash that
nvcc derives from the translation unit path. That check is a necessary but not
sufficient gate: constants that happen to share a value at one configuration
alias in the compiled output, so the correctness matrix must be rerun for every
configuration. Two such aliases were found and fixed while building this family
(the per-epoch expert-array reset and the combine-path owner derivation both
read as `32` only because the default local expert count is 32).

The earlier G8 geometry keeps one intentional difference: its padded-row guard
is now `kMaxTasks * kTaskM` (101376) instead of the previous 101328. Both are
sound upper bounds, and the tighter original could never be reached, but the
derived bound is the one that matches the `padded == tasks * kTaskM` audit.
