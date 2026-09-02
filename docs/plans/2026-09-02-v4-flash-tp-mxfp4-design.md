# DeepSeek-V4-Flash MXFP4 TP-MoE benchmark and optimization design

Date: 2026-09-02

## Scope

Optimize the previously reviewed SM90 WGMMA MXFP4 kernels in `DeepGEMM_tp` for the routed-expert portion of DeepSeek-V4-Flash under tensor parallelism. TP4 is the primary target; TP8 must compile, pass correctness, and run end to end. This is not an EP benchmark and does not include the router/top-k calculation or the shared expert.

The model configuration is:

- hidden size `H=4096`
- routed intermediate size `I=2048`
- `E=256` routed experts
- `topk=6`
- SwiGLU activation

Per-rank TP shapes are:

| TP | I shard | W13 / FC1 | W2 / FC2 |
|---:|---:|---|---|
| 4 | 512 | `[256, 1024, 4096]` | `[256, 4096, 512]` |
| 8 | 256 | `[256, 512, 4096]` | `[256, 4096, 256]` |

## Data contract and semantics

Each rank receives the same:

- BF16 `X [M,4096]`
- precomputed INT32 `topk_idx [M,6]`
- precomputed FP32 `topk_weights [M,6]`

Each route must read `X[token_id]` and expert-specific MXFP4 weights. The old raw-throughput `G` knob is not a proxy for tokens, active experts, or routing and is excluded from all verdicts.

Experts are replicated logically across TP ranks while their intermediate dimension is sharded. A rank computes its local SwiGLU intermediate slice and W2 partial output, reduces the six weighted route results locally into BF16 `Y_partial [M,4096]`, and participates in one TP sum all-reduce. There is no EP dispatch, remote expert ownership, or EP return/combine phase.

## Formal Humming baseline

The baseline follows SGLang's standard-dispatch Humming indexed runner. Standard dispatch/combine are pass-through in this pure-TP configuration. The timed CUDA work is:

1. `moe_align_block_size` route alignment for indexed GEMM.
2. Dynamic BF16-to-FP8-E4M3 input quantization with group size 128.
3. Humming OCP-MXFP4 W13 GEMM: E2M1 weights with E8M0 group-32 scales.
4. SwiGLU.
5. Dynamic FP8-E4M3 group-128 quantization of W2 input.
6. Humming OCP-MXFP4 W2 GEMM.
7. SGLang local k=6 weighted reduction (`moe_fused_mul_sum`) to `[M,4096]`.
8. Exactly one SGLang `CustomAllReduceV2` over the BF16 output.

The baseline must use SGLang's normal `CustomAllReduceV2` selection heuristics, without forcing an algorithm or lifting crossover thresholds. The formal benchmark is fixed-shape CUDA Graph replay; graph capture, JIT, weight conversion, allocation, and top-k generation are excluded. Eager runs are debugging evidence only.

## Measurements and score

Required M values are `8, 16, 32, 64, 128`. For each point:

- synchronize ranks before an outer sample;
- time the graph replay on every rank;
- use the maximum rank latency for that sample;
- run at least five outer samples and report min, median, and max;
- retain stage-level timings outside the formal score to explain wins or regressions.

The primary tuning score is the equal-weight geometric mean of the five TP4 max-rank median latencies. Every individual M result remains visible; a geometric-mean win cannot hide a serious point regression. TP8 has the same five-M run-through and correctness requirements but is not used to choose TP4 tuning.

Two routing families are required: deterministic uniform-without-replacement routes and a deterministic hot-expert/skewed case. Both use real token IDs, expert IDs, route weights, and expert-specific weights. If a production V4-Flash routing trace is available, add it as a third family without replacing the synthetic controls.

## Correctness

Generate one full quantized model's logical weights, shard I identically for TP4/TP8, and transform the shards separately into Humming and candidate layouts. Compare both against a torch reference that dequantizes the exact packed MXFP4 values, applies the same FP8 activation quantization, SwiGLU/clamp semantics, route weights, local sum, and TP sum.

Correctness gates include finite output, per-token cosine, norm ratio, absolute-error statistics, correct handling of inactive experts and skewed routes, and agreement across TP4 and TP8. Humming is a performance baseline, not the numerical oracle.

## Candidate implementation path

Start from the reviewed `step_e_lutg.py` FC1 WGMMA kernel, `step_e_fc2.py` FC2 WGMMA kernel, and the existing TP/symmetric-memory work in this worktree.

1. Replace `G`-only tile generation with a GPU route-alignment table containing token ID, expert ID, route slot, valid rows, and expert offsets. Preserve expert-specific packed weights and gather the correct X row for every route.
2. First establish a correct two-compute-kernel TP pipeline: route-aware W13 with fused SwiGLU and FP8 requantization, route-aware W2, local weighted reduction, then unmodified `CustomAllReduceV2`.
3. Reuse the DeepGEMM MegaMoE persistent/interleaved scheduling ideas while deleting EP pull/scatter/barrier logic. Allow TP4 and TP8 shape specializations from the same source.
4. Only after the separate-compute plus stock-all-reduce version is correct and faster, investigate producer handoff or overlap with `CustomAllReduceV2`. Keep the stock SGLang path as the immutable baseline.

Every performance modification is one logged iteration: edit, run the formal or explicitly marked diagnostic benchmark, append the result and conclusion to `ITERATIONS.md`, then commit only the files owned by this effort. The dirty DeepGEMM main checkout remains read-only.
