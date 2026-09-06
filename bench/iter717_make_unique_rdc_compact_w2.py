#!/usr/bin/env python3
"""Create a unique RDC build with a one-pointer M128 W2 phase ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


MODULE = "v4tp_49f35bd1b995105c53bc_v178mspec"
HOST_ENTRY = "run_tp4_megamoe_single_launch"
CUDA_ENTRY = "tp4_megamoe_single_launch_kernel"


def rewrite(path: Path, old: str, new: str, expected: int) -> None:
    text = path.read_text()
    actual = text.count(old)
    if actual != expected:
        raise RuntimeError(
            f"{path}: expected {expected} occurrences of {old!r}, got {actual}"
        )
    path.write_text(text.replace(old, new))


def insert_before(path: Path, marker: str, payload: str) -> None:
    text = path.read_text()
    actual = text.count(marker)
    if actual != 1:
        raise RuntimeError(
            f"{path}: expected one insertion marker, got {actual}"
        )
    path.write_text(text.replace(marker, payload + marker))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-tag", default="iter714b")
    parser.add_argument("--tag", default="iter717a")
    args = parser.parse_args()

    for value, name in ((args.source_tag, "--source-tag"), (args.tag, "--tag")):
        if not value.replace("_", "").isalnum():
            raise ValueError(f"{name} must be alphanumeric/underscore")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    for name in ("cuda.cu", "main.cpp", "build.ninja"):
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)

    old_host = f"{HOST_ENTRY}_{args.source_tag}"
    old_cuda = f"{CUDA_ENTRY}_{args.source_tag}"
    new_host = f"{HOST_ENTRY}_{args.tag}"
    new_cuda = f"{CUDA_ENTRY}_{args.tag}"
    compact_callee = f"single_launch_w2_gemm_phase_compact_rdc_{args.tag}"
    cuda_source = output_dir / "cuda.cu"
    host_source = output_dir / "main.cpp"
    build_file = output_dir / "build.ninja"

    rewrite(cuda_source, old_cuda, new_cuda, 5)
    rewrite(cuda_source, old_host, new_host, 1)
    rewrite(host_source, old_host, new_host, 2)
    rewrite(build_file, str(source_dir), str(output_dir), 2)

    compact_definition = f"""
// Iteration 717: keep the RDC phase boundary but collapse its M128 call ABI
// to one CTA-shared pointer.  The callee derives every uniform scalar from
// its shape specialization and the published record.
template <int Tokens, bool AssumeValidMblock>
__device__ __noinline__ void {compact_callee}(
        const SingleLaunchW13PhaseArgs* __restrict__ args) {{
    constexpr int kW2NTiles = 4096 / kWout;
    constexpr int kMaxRoutes = Tokens * kTopK;
    const int cta = static_cast<int>(blockIdx.x);
    const int ctas = static_cast<int>(gridDim.x);
    const int tasks =
        (__ldg(args->num_tokens_padded) / kTok) * kW2NTiles;
    for (int task = cta; task < tasks; task += ctas) {{
        route_gemm_task<
            512, 4096, 1, false, 0, false, false, false, -1, false, 0,
            AssumeValidMblock>(
            args->tma_weight, args->tma_weight_scale,
            args->weight, args->weight_scale, args->weight_global_scale,
            args->activation, args->activation_scale,
            args->sorted_ids, args->expert_ids, args->num_tokens_padded,
            args->topk_weights, args->output, args->global_lut, nullptr,
            kMaxRoutes, 0, task);
        __syncthreads();
    }}
}}

"""
    insert_before(
        cuda_source,
        "// Keep the selected standalone launch as a thin wrapper around the task body.\n",
        compact_definition,
    )

    old_call = """            } else if constexpr (kSingleLaunchW2PhaseNoInline) {
                single_launch_w2_gemm_phase<
                    kSingleLaunchAssumeValidGemmTasks>(
                    &w2_tma_weight, &w2_tma_weight_scale,
                    w2, s2, g2, qactivation, activation_scale,
                    sorted_ids, expert_ids, num_tokens_padded,
                    topk_weights, down, lut, routes,
                    cta, ctas, w2_tasks);
            } else {
"""
    new_call = f"""            }} else if constexpr (kSingleLaunchW2PhaseNoInline) {{
                if constexpr (Tokens == 128) {{
                    // W13 is complete and this record is dead until the next
                    // replay, so republish the W2-uniform arguments once.
                    if (threadIdx.x == 0) {{
                        w13_phase_args.tma_weight = &w2_tma_weight;
                        w13_phase_args.tma_weight_scale =
                            &w2_tma_weight_scale;
                        w13_phase_args.weight = w2;
                        w13_phase_args.weight_scale = s2;
                        w13_phase_args.weight_global_scale = g2;
                        w13_phase_args.activation = qactivation;
                        w13_phase_args.activation_scale = activation_scale;
                        w13_phase_args.sorted_ids = sorted_ids;
                        w13_phase_args.expert_ids = expert_ids;
                        w13_phase_args.num_tokens_padded = num_tokens_padded;
                        w13_phase_args.topk_weights = topk_weights;
                        w13_phase_args.output =
                            reinterpret_cast<float*>(down);
                        w13_phase_args.global_lut = lut;
                    }}
                    __syncthreads();
                    {compact_callee}<
                        Tokens, kSingleLaunchAssumeValidGemmTasks>(
                        &w13_phase_args);
                }} else {{
                    single_launch_w2_gemm_phase<
                        kSingleLaunchAssumeValidGemmTasks>(
                        &w2_tma_weight, &w2_tma_weight_scale,
                        w2, s2, g2, qactivation, activation_scale,
                        sorted_ids, expert_ids, num_tokens_padded,
                        topk_weights, down, lut, routes,
                        cta, ctas, w2_tasks);
                }}
            }} else {{
"""
    rewrite(cuda_source, old_call, new_call, 1)

    print(
        json.dumps(
            {
                "compact_callee": compact_callee,
                "cuda_entry": new_cuda,
                "cuda_sha256": sha256(cuda_source),
                "host_entry": new_host,
                "host_sha256": sha256(host_source),
                "module": MODULE,
                "ninja_sha256": sha256(build_file),
                "output_dir": str(output_dir),
                "source_dir": str(source_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
