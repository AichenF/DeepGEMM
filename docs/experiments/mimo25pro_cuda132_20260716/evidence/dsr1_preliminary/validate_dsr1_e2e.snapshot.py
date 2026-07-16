#!/usr/bin/env python3
"""Strict, dependency-free validator for the frozen DeepSeek-R1 GSM8K E2E run."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DG_COMMIT = "75186dde9dac140c053c9007ace0ce7cce41150c"
SGLANG_COMMIT = "b6a68c9acb6590b2849febe2b66807553923fc71"
SGLANG_TAR_SHA256 = "258ed9c1538dd534f6cbd0781b8c559ff99ea934b49b24940c96a279c8af1984"
IMAGE_DIGEST = "sha256:5027e95bf6ec536856b1b52a91d1f35ff5c564ab83e8a94758a169ff09bb8df3"
IMAGE_REF = "lmsysorg/sglang@" + IMAGE_DIGEST
GSM8K_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
PREREGISTERED_PROTOCOL_SHA256 = "7444b1bc992b98299449822991bbdc18ca4d7487359dc0377fffa0d95c2dfd7f"
GSM8K_TOTAL_ROWS = 1319
NUM_QUESTIONS = 1316
NUM_SHOTS = 8
PARALLEL = 1316
MAX_NEW_TOKENS = 512
MIN_REPORTED_ACCURACY = 0.957
# Do not let three-decimal display rounding turn 1259/1316 (0.956687) into a
# false pass.  The formal gate is on the independently recomputed exact ratio.
MIN_CORRECT = math.ceil(MIN_REPORTED_ACCURACY * NUM_QUESTIONS)
MODEL_SHARDS = 163
MODEL_BYTES = 413_328_348_544
MODEL_PATH = "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-0528-FP4-v2"
EXPECTED_MOE_LAYERS = 58
INVALID = -9_999_999

MODEL_METADATA_SHA256 = {
    "config.json": "a80de10ccf70e7e98dcc6730b45872298937b232d3e8eb0321cfd2bce4cb40e0",
    "hf_quant_config.json": "53c65ef081fa434e6b75a39a50ac9f15e613c880f3a0b65119c6c845156f33ca",
    "model.safetensors.index.json": "72c46aec135fca6cd4a7338874e687c35a4ee8e83bc03c7f2114de42954e6e51",
    "tokenizer.json": "ecb6f9fc369894346f0511f4074ca75cee5cd5f3b06d02f1ba35fcd39f8e121d",
    "tokenizer_config.json": "064a07ba2935e4caa8bff563188e3f59310dbf2c0781559dc0ad4f722edbe6d2",
    "generation_config.json": "0ef9febae6b6087f4822b02bc9a1c03a83263dabaa931fb9155d061c8951ca07",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValidationError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            require(isinstance(row, dict), f"{path}:{lineno}: expected JSON object")
            rows.append(row)
    return rows


def answer_value(text: str) -> int | float:
    text = text.replace(",", "")
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return INVALID
    try:
        value = ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID
    return value if isinstance(value, (int, float)) else INVALID


def one_example(rows: list[dict[str, Any]], index: int, include_answer: bool) -> str:
    text = "Question: " + rows[index]["question"] + "\nAnswer:"
    if include_answer:
        text += " " + rows[index]["answer"]
    return text


def expected_prompts(dataset: list[dict[str, Any]]) -> list[str]:
    few_shot = "".join(one_example(dataset, i, True) + "\n\n" for i in range(NUM_SHOTS))
    return [few_shot + one_example(dataset, i, False) for i in range(NUM_QUESTIONS)]


def corruption_reasons(output: str) -> list[str]:
    reasons: list[str] = []
    if "\ufffd" in output or "\x00" in output:
        reasons.append("replacement_or_nul")
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in output):
        reasons.append("control_character")
    collapsed = re.sub(r"\s+", " ", output)
    # Known race failures produced long repeated digit/word/code fragments.
    if re.search(r"(?i)([^\s]{2,32})(?:\s*\1){7,}", collapsed):
        reasons.append("repeated_fragment")
    if re.search(r"([{}:;,])(?:\s*\1){31,}", collapsed):
        reasons.append("repeated_punctuation")
    return reasons


def validate_dataset(path: Path, *, enforce_pinned_hash: bool = True) -> dict[str, Any]:
    require(path.is_file(), f"missing dataset: {path}")
    observed_sha = sha256_file(path)
    if enforce_pinned_hash:
        require(observed_sha == GSM8K_SHA256, f"GSM8K SHA mismatch: {observed_sha}")
    rows = load_jsonl(path)
    require(len(rows) == GSM8K_TOTAL_ROWS, f"expected {GSM8K_TOTAL_ROWS} dataset rows, got {len(rows)}")
    for index, row in enumerate(rows):
        require(set(row) >= {"question", "answer"}, f"dataset row {index} lacks question/answer")
        require(isinstance(row["question"], str) and row["question"], f"empty question {index}")
        require(isinstance(row["answer"], str) and row["answer"], f"empty answer {index}")
    labels = [answer_value(row["answer"]) for row in rows[:NUM_QUESTIONS]]
    require(all(value != INVALID for value in labels), "dataset contains an invalid label")
    return {
        "sha256": observed_sha,
        "source_commit": GSM8K_COMMIT,
        "rows": len(rows),
        "evaluated_rows": NUM_QUESTIONS,
        "shots": NUM_SHOTS,
        "leakage_disclosure": "the first 8 test rows are both demonstrations and scored rows",
    }


def validate_model(model_dir: Path) -> dict[str, Any]:
    require(model_dir.is_dir(), f"missing model directory: {model_dir}")
    shards = sorted(model_dir.glob("model-*-of-000163.safetensors"))
    expected_names = [f"model-{i:05d}-of-000163.safetensors" for i in range(1, MODEL_SHARDS + 1)]
    require([path.name for path in shards] == expected_names, "model shard names/count are not exactly 1..163")
    sizes = {path.name: path.stat().st_size for path in shards}
    require(all(size > 0 for size in sizes.values()), "one or more model shards are empty")
    require(sum(sizes.values()) == MODEL_BYTES, f"model byte total mismatch: {sum(sizes.values())}")
    observed_meta: dict[str, str] = {}
    for name, expected in MODEL_METADATA_SHA256.items():
        path = model_dir / name
        require(path.is_file(), f"missing model metadata: {name}")
        observed_meta[name] = sha256_file(path)
        require(observed_meta[name] == expected, f"model metadata SHA mismatch: {name}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "hidden_size": 7168,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 61,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
    }
    for key, value in expected_config.items():
        require(config.get(key) == value, f"model config {key}={config.get(key)!r}, expected {value!r}")
    quant = json.loads((model_dir / "hf_quant_config.json").read_text(encoding="utf-8"))["quantization"]
    require(quant.get("quant_algo") == "NVFP4", "model is not NVFP4")
    require(quant.get("kv_cache_quant_algo") == "FP8", "model KV cache is not FP8")
    require(quant.get("group_size") == 16, "model group_size is not 16")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    referenced = sorted(set(index.get("weight_map", {}).values()))
    require(referenced == expected_names, "model index does not reference exactly all 163 shards")
    return {
        "model_dir": str(model_dir),
        "shard_count": len(shards),
        "shard_bytes": sum(sizes.values()),
        "shard_sizes": sizes,
        "metadata_sha256": observed_meta,
        "config": expected_config,
        "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16},
    }


@dataclass
class RoundSummary:
    name: str
    correct: int
    total: int
    accuracy_exact: float
    accuracy_reported: float
    invalid: int
    corrupt_outputs: int
    latency_s: float
    output_throughput_tok_s: float
    result_num_gpus_field: int
    pass_status: bool
    raw_sha256: str
    result_sha256: str
    benchmark_log_sha256: str


def parse_metric(log_text: str, label: str) -> float:
    matches = re.findall(rf"(?m)^{re.escape(label)}:\s*([0-9]+(?:\.[0-9]+)?)", log_text)
    require(len(matches) == 1, f"benchmark log must contain exactly one {label} line, got {len(matches)}")
    return float(matches[0])


def validate_round(run_dir: Path, name: str, dataset: list[dict[str, Any]]) -> RoundSummary:
    result_path = run_dir / f"gsm8k_result_{name}.jsonl"
    raw_path = run_dir / f"gsm8k_raw_{name}.jsonl"
    log_path = run_dir / f"benchmark_{name}.log"
    for path in (result_path, raw_path, log_path):
        require(path.is_file(), f"missing {name} artifact: {path.name}")
    result_rows = load_jsonl(result_path)
    require(len(result_rows) == 1, f"{result_path.name} must contain exactly one row")
    result = result_rows[0]
    require(result.get("task") == "gsm8k", f"unexpected task in {result_path.name}")
    require(result.get("backend") == "srt", f"unexpected backend in {result_path.name}")
    # This client field is hard-coded to one. It is never accepted as service-rank evidence.
    require(result.get("num_gpus") == 1, "unexpected client num_gpus field; protected harness may have changed")
    require(result.get("num_requests") == NUM_QUESTIONS, "result num_requests mismatch")
    require(result.get("other", {}).get("num_questions") == NUM_QUESTIONS, "result num_questions mismatch")
    require(result.get("other", {}).get("parallel") == PARALLEL, "result parallel mismatch")

    raw = load_jsonl(raw_path)
    require(len(raw) == NUM_QUESTIONS, f"{raw_path.name}: expected {NUM_QUESTIONS} rows, got {len(raw)}")
    prompts = expected_prompts(dataset)
    labels = [answer_value(row["answer"]) for row in dataset[:NUM_QUESTIONS]]
    correct = invalid = corrupt = 0
    corruption_examples: list[str] = []
    for index, row in enumerate(raw):
        require(row.get("prompt_id") == index, f"{raw_path.name}: prompt_id mismatch at row {index}")
        require(row.get("prompt") == prompts[index], f"{raw_path.name}: prompt mismatch at row {index}")
        output = row.get("output")
        require(isinstance(output, str), f"{raw_path.name}: non-string output at row {index}")
        prediction = answer_value(output)
        recomputed = prediction == labels[index]
        require(row.get("correct") is recomputed, f"{raw_path.name}: stored correct flag mismatch at row {index}")
        correct += int(recomputed)
        invalid += int(prediction == INVALID)
        reasons = corruption_reasons(output)
        if reasons:
            corrupt += 1
            if len(corruption_examples) < 10:
                corruption_examples.append(f"prompt_id={index}:{','.join(reasons)}")
    require(invalid == 0, f"{name}: invalid outputs={invalid}")
    require(corrupt == 0, f"{name}: corruption signatures={corrupt}: {corruption_examples}")
    accuracy_exact = correct / NUM_QUESTIONS
    accuracy_reported = round(accuracy_exact, 3)
    require(correct >= MIN_CORRECT, f"{name}: correct={correct} below {MIN_CORRECT}")
    require(accuracy_exact >= MIN_REPORTED_ACCURACY, f"{name}: exact accuracy={accuracy_exact:.9f}")
    require(accuracy_reported >= MIN_REPORTED_ACCURACY, f"{name}: reported accuracy={accuracy_reported:.3f}")

    require(abs(float(result.get("accuracy")) - accuracy_reported) < 5e-7, "result accuracy disagrees with raw")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    log_acc = parse_metric(log_text, "Accuracy")
    log_invalid = parse_metric(log_text, "Invalid")
    latency = parse_metric(log_text, "Latency")
    throughput = parse_metric(log_text, "Output throughput")
    require(abs(log_acc - accuracy_reported) < 5e-7, "benchmark Accuracy disagrees with raw")
    require(log_invalid == 0.0, "benchmark Invalid is nonzero")
    require(math.isfinite(latency) and latency > 0, "invalid benchmark latency")
    require(math.isfinite(throughput) and throughput > 0, "invalid benchmark throughput")
    require(abs(float(result.get("latency")) - latency) < 5e-4, "result latency disagrees with benchmark log")
    return RoundSummary(
        name=name,
        correct=correct,
        total=NUM_QUESTIONS,
        accuracy_exact=accuracy_exact,
        accuracy_reported=accuracy_reported,
        invalid=invalid,
        corrupt_outputs=corrupt,
        latency_s=latency,
        output_throughput_tok_s=throughput,
        result_num_gpus_field=1,
        pass_status=True,
        raw_sha256=sha256_file(raw_path),
        result_sha256=sha256_file(result_path),
        benchmark_log_sha256=sha256_file(log_path),
    )


def parse_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [[cell.strip() for cell in row] for row in csv.reader(handle) if row]


def validate_service_evidence(run_dir: Path) -> dict[str, Any]:
    gpu_path = run_dir / "gpu_before.csv"
    process_path = run_dir / "service_gpu_processes.csv"
    top_path = run_dir / "service_server_processes.txt"
    identity_path = run_dir / "server_identity.log"
    server_log_path = run_dir / "server.log"
    server_inspect_path = run_dir / "server_inspect.json"
    command_path = run_dir / "server_command.txt"
    extension_path = run_dir / "deepgemm_extension.sha256"
    for path in (gpu_path, process_path, top_path, identity_path, server_log_path, server_inspect_path, command_path, extension_path):
        require(path.is_file(), f"missing service evidence: {path.name}")
    gpu_rows = parse_csv_rows(gpu_path)
    require(len(gpu_rows) == 9, "gpu_before.csv must contain header + 8 GPUs")
    header = gpu_rows[0]
    require(header[:3] == ["index", "name", "uuid"], f"unexpected GPU header: {header}")
    gpu_uuids = {row[2] for row in gpu_rows[1:]}
    require(len(gpu_uuids) == 8, "expected eight unique GPU UUIDs")
    require(all("H200" in row[1] for row in gpu_rows[1:]), "not all GPUs are H200")
    process_rows = parse_csv_rows(process_path)
    require(process_rows, "service_gpu_processes.csv is empty")
    for row in process_rows:
        require(len(row) >= 3, f"malformed service process row: {row}")
        require(row[0] in gpu_uuids, f"unexpected service GPU UUID: {row[0]}")
        require(row[1].isdigit(), f"non-numeric service GPU PID: {row[1]}")
        # NVML may report N/A for per-process memory on some driver/container
        # combinations.  UUID and host PID are the ownership evidence; memory
        # is retained as useful telemetry but is not a correctness field.
        require(row[2].isdigit() or row[2] == "N/A", f"invalid service GPU memory: {row[2]}")
    process_uuids = {row[0] for row in process_rows}
    gpu_pids = {row[1] for row in process_rows}
    require(process_uuids == gpu_uuids, f"service does not occupy all eight GPU UUIDs: {process_uuids}")
    require(len(gpu_pids) >= 8, "fewer than eight service GPU PIDs")
    top_lines = top_path.read_text(encoding="utf-8", errors="replace").splitlines()
    require(top_lines and top_lines[0].split()[0] == "PID", "unexpected docker top header")
    container_pids = {
        line.split()[0]
        for line in top_lines[1:]
        if line.split() and line.split()[0].isdigit()
    }
    require(gpu_pids <= container_pids, f"GPU PIDs are not all owned by the server container: {sorted(gpu_pids - container_pids)}")

    identity = identity_path.read_text(encoding="utf-8", errors="replace")
    required_identity = [
        f"DEEPGEMM_COMMIT={DG_COMMIT}",
        "DEEPGEMM_PATH=/workspace/DeepGEMM/deep_gemm/__init__.py",
        "DEEPGEMM_EXTENSION_PATH=/workspace/DeepGEMM/deep_gemm/_C",
        "SGLANG_PATH=/workspace/sglang/python/sglang/__init__.py",
        "TORCH_CUDA=13.0",
        "CUDA_DEVICE_COUNT=8",
        "GSM8K_HARNESS_SHA256=252f15aec6f1275eacf6bb58eaedb4198b3cf9c70af2f220d94495d0defb70e6",
    ]
    for marker in required_identity:
        require(marker in identity, f"server identity lacks {marker}")
    extension_manifest = extension_path.read_text(encoding="utf-8").strip()
    extension_match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", extension_manifest)
    require(extension_match is not None, "malformed DeepGEMM extension manifest")
    extension_sha = extension_match.group(1)
    require(
        f"DEEPGEMM_EXTENSION_SHA256={extension_sha}" in identity,
        "server imported a different DeepGEMM extension",
    )
    server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
    require("The server is fired up and ready to roll!" in server_log, "server ready marker missing")
    built_lines = [line for line in server_log.splitlines() if "Built sm90_nvfp4 MegaMoE weights" in line]
    require(len(built_lines) == 8 * EXPECTED_MOE_LAYERS, f"expected {8 * EXPECTED_MOE_LAYERS} SM90 NVFP4 layer-build lines, got {len(built_lines)}")
    require(all("block_n=256" in line and "grouped_nibbles=True" in line for line in built_lines), "server built a non-BN256/non-grouped layer")
    layer_ids: list[int] = []
    rank_layers: dict[tuple[int, int, int], list[int]] = {}
    for line in built_lines:
        match = re.search(r"DP(\d+) TP(\d+) EP(\d+)\].*Built sm90_nvfp4 MegaMoE weights for layer (\d+)\s*\(", line)
        require(match is not None, f"cannot parse SM90 NVFP4 layer id: {line}")
        rank_key = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        layer_id = int(match.group(4))
        layer_ids.append(layer_id)
        rank_layers.setdefault(rank_key, []).append(layer_id)
    expected_layer_ids = set(range(3, 61))
    require(
        set(layer_ids) == expected_layer_ids,
        f"SM90 NVFP4 unique layer ids mismatch: {sorted(set(layer_ids))}",
    )
    expected_rank_keys = {(rank, rank, rank) for rank in range(8)}
    require(set(rank_layers) == expected_rank_keys, f"unexpected DP/TP/EP build rank tuples: {sorted(rank_layers)}")
    for rank_key, observed_layers in sorted(rank_layers.items()):
        require(len(observed_layers) == EXPECTED_MOE_LAYERS, f"rank {rank_key} has {len(observed_layers)} layer-build lines")
        require(set(observed_layers) == expected_layer_ids, f"rank {rank_key} layer IDs mismatch")
    lowered = server_log.lower()
    for bad in ("cuda out of memory", "nccl error", "watchdog timeout", "runtimeerror", "traceback"):
        require(bad not in lowered, f"server log contains {bad!r}")

    server_arg_lines = [line for line in server_log.splitlines() if "server_args=ServerArgs(" in line]
    require(len(server_arg_lines) == 1, f"expected one actual ServerArgs line, got {len(server_arg_lines)}")
    actual_args = server_arg_lines[0]
    required_actual_args = [
        "quantization='modelopt_fp4'", "tp_size=8", "dp_size=8", "ep_size=8",
        "enable_dp_attention=True", "moe_a2a_backend='megamoe'",
        "disable_cuda_graph=True", "disable_radix_cache=True",
        "skip_server_warmup=True", "max_running_requests=1316",
        "watchdog_timeout=900.0", "mem_fraction_static=0.9", "enable_eplb=False",
    ]
    for marker in required_actual_args:
        require(marker in actual_args, f"actual ServerArgs lacks {marker}")

    inspect_value = json.loads(server_inspect_path.read_text(encoding="utf-8"))
    require(isinstance(inspect_value, list) and len(inspect_value) == 1, "malformed server inspect JSON")
    inspect_config = inspect_value[0].get("Config", {})
    require(inspect_config.get("Image") == IMAGE_REF, f"actual server image mismatch: {inspect_config.get('Image')}")
    actual_cmd = " ".join(str(part) for part in inspect_config.get("Cmd", []))
    actual_env_rows = inspect_config.get("Env", [])
    require(isinstance(actual_env_rows, list), "actual server Env is malformed")
    actual_env = {row.split("=", 1)[0]: row.split("=", 1)[1] for row in actual_env_rows if isinstance(row, str) and "=" in row}
    required_env = {
        "DG_SM90_NVFP4_BLOCK_N": "256",
        "DG_SM90_NVFP4_NIBBLE_GROUP": "1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
        "DG_JIT_CACHE_DIR": "/out/dg_jit_cache",
        "PYTHONPATH": "/workspace/sglang/python:/workspace/DeepGEMM",
    }
    for key, value in required_env.items():
        require(actual_env.get(key) == value, f"actual server Env mismatch for {key}")
    for forbidden in ("SGLANG_MEGAMOE_NVFP4_REQUANTIZE", "DG_SM90_NVFP4_FUSED_B_SCALE"):
        require(forbidden not in actual_env, f"forbidden actual server Env is present: {forbidden}")

    command = command_path.read_text(encoding="utf-8")
    required_cli_command = [
        f"--model-path {MODEL_PATH}",
        "--quantization modelopt_fp4", "--tp 8", "--dp 8", "--enable-dp-attention",
        "--moe-a2a-backend megamoe", "--disable-cuda-graph", "--disable-radix-cache",
        "--skip-server-warmup", "--max-running-requests 1316", "--watchdog-timeout 900",
        "--mem-fraction-static 0.9",
    ]
    required_command_env = [f"{key}={value}" for key, value in required_env.items()]
    for token in required_cli_command + required_command_env:
        require(token in command, f"server command lacks {token}")
    require("--chunked-prefill-size" not in command, "chunked-prefill-size must not be passed")
    # Docker records -e values in Config.Env, not Config.Cmd.  Requiring the
    # environment assignments in actual_cmd would reject every correct launch.
    for token in required_cli_command:
        require(token in actual_cmd, f"actual Docker Cmd lacks {token}")
    require("--chunked-prefill-size" not in actual_cmd, "actual Docker Cmd must not pass chunked-prefill-size")
    return {
        "service_gpu_count": 8,
        "service_gpu_uuid_count": len(process_uuids),
        "service_pid_count": len(gpu_pids),
        "server_container_pid_count": len(container_pids),
        "all_gpu_pids_owned_by_server_container": True,
        "deepgemm_extension_sha256": extension_sha,
        "sm90_nvfp4_layer_build_lines": len(built_lines),
        "sm90_nvfp4_unique_layer_count": len(set(layer_ids)),
        "sm90_nvfp4_unique_layer_ids": sorted(set(layer_ids)),
        "sm90_nvfp4_complete_rank_count": len(rank_layers),
        "eplb_disabled_in_actual_server_args": True,
        "docker_image_cmd_env_verified": True,
        "client_result_num_gpus_field_is_not_service_evidence": True,
    }


def read_single_line(path: Path) -> str:
    require(path.is_file(), f"missing identity file: {path.name}")
    return path.read_text(encoding="utf-8").strip()


def validate_jit_evidence(run_dir: Path) -> dict[str, Any]:
    warmup_path = run_dir / "jit_cache_after_warmup.sha256"
    timed_path = run_dir / "jit_cache_after_timed.sha256"
    grouped_path = run_dir / "grouped_nibble_jit_entries.txt"
    for path in (warmup_path, timed_path, grouped_path):
        require(path.is_file(), f"missing JIT evidence: {path.name}")
    warmup_text = warmup_path.read_text(encoding="utf-8")
    timed_text = timed_path.read_text(encoding="utf-8")
    require(warmup_text.strip() != "", "warmup JIT manifest is empty")
    require(warmup_text == timed_text, "DeepGEMM JIT cache changed during timed round")
    expected_grouped = [
        line
        for line in warmup_text.splitlines()
        if "kernel.sm90_nvfp4_mega_moe_nibble_group." in line
    ]
    observed_grouped = grouped_path.read_text(encoding="utf-8").splitlines()
    require(expected_grouped, "JIT manifest has no grouped-nibble MegaMoE entry")
    require(observed_grouped == expected_grouped, "grouped JIT evidence does not match manifest")
    require(any(line.endswith("/kernel.cubin") for line in expected_grouped), "grouped JIT evidence has no cubin")
    variants = {
        line.split("kernel.sm90_nvfp4_mega_moe_nibble_group.", 1)[1].split("/", 1)[0]
        for line in expected_grouped
        if "/cache/kernel.sm90_nvfp4_mega_moe_nibble_group." in line
        and line.endswith("/kernel.cubin")
    }
    return {
        "manifest_entries": len(warmup_text.splitlines()),
        "grouped_manifest_entries": len(expected_grouped),
        "grouped_variant_count": len(variants),
        "unchanged_during_timed_round": True,
    }


def summarize_run(run_dir: Path, *, enforce_pinned_dataset: bool = True, write_outputs: bool = True) -> dict[str, Any]:
    require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    require(read_single_line(run_dir / "deepgemm_commit.txt") == DG_COMMIT, "DeepGEMM identity mismatch")
    require(read_single_line(run_dir / "sglang_commit.txt") == SGLANG_COMMIT, "SGLang identity mismatch")
    require(read_single_line(run_dir / "sglang_source_sha256.txt") == SGLANG_TAR_SHA256, "SGLang tar identity mismatch")
    require(read_single_line(run_dir / "container_image_digest.txt") == IMAGE_DIGEST, "container digest mismatch")
    protocol_snapshot = run_dir / "protocol_preregistered.snapshot.md"
    require(protocol_snapshot.is_file(), "missing preregistered protocol snapshot")
    require(sha256_file(protocol_snapshot) == PREREGISTERED_PROTOCOL_SHA256, "preregistered protocol hash mismatch")
    dataset_path = run_dir / "gsm8k_test.jsonl"
    dataset_info = validate_dataset(dataset_path, enforce_pinned_hash=enforce_pinned_dataset)
    dataset = load_jsonl(dataset_path)
    model_inventory = json.loads((run_dir / "model_inventory.json").read_text(encoding="utf-8"))
    require(model_inventory.get("shard_count") == MODEL_SHARDS, "model inventory shard count mismatch")
    require(model_inventory.get("shard_bytes") == MODEL_BYTES, "model inventory byte total mismatch")
    service = validate_service_evidence(run_dir)
    jit = validate_jit_evidence(run_dir)
    warmup = validate_round(run_dir, "warmup", dataset)
    timed = validate_round(run_dir, "timed", dataset)
    require(read_single_line(run_dir / "round_order.txt") == "warmup(excluded)->timed(formal)", "round order mismatch")
    require(read_single_line(run_dir / "retry_policy.txt") == "no_score_retry_or_replacement", "retry policy mismatch")
    summary = {
        "status": "PASS",
        "scope": "DeepSeek-R1 full-model SGLang GSM8K E2E accuracy proxy",
        "deepgemm_commit": DG_COMMIT,
        "sglang_commit": SGLANG_COMMIT,
        "container_image_digest": IMAGE_DIGEST,
        "container_cuda": "13.0.1 / torch cu130 (not CUDA 13.2)",
        "dataset": dataset_info,
        "model": {"shards": MODEL_SHARDS, "bytes": MODEL_BYTES, "geometry": "H7168/I2048/E256/topk8", "layers": 61, "moe_layers": 58},
        "service": service,
        "jit": jit,
        "warmup_excluded_from_timing": asdict(warmup),
        "timed_formal": asdict(timed),
        "acceptance": {"min_reported_accuracy": MIN_REPORTED_ACCURACY, "min_correct": MIN_CORRECT, "invalid_required": 0, "corruption_signatures_required": 0},
        "interpretation": "result JSON num_gpus=1 is a protected client-side constant; eight-rank proof comes from service GPU UUID/PID evidence",
    }
    if write_outputs:
        (run_dir / "e2e_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md = [
            "# DeepSeek-R1 最终版 E2E 验证摘要", "", "状态：**PASS**", "",
            f"- 最终组合：DeepGEMM `{DG_COMMIT[:8]}` + SGLang `{SGLANG_COMMIT[:8]}`",
            f"- 正式轮：{timed.correct}/{timed.total}，accuracy `{timed.accuracy_reported:.3f}`，invalid `{timed.invalid}`，异常重复/乱码 `{timed.corrupt_outputs}`",
            f"- 正式轮耗时：`{timed.latency_s:.3f}s`，输出吞吐：`{timed.output_throughput_tok_s:.3f} token/s`",
            "- 服务证据：8 个 H200 UUID 均有服务进程；JSON 中 `num_gpus=1` 是客户端硬编码字段，不用于判断服务卡数。",
            "- 环境边界：固定 SGLang 镜像为 CUDA 13.0.1/torch cu130，不是 CUDA 13.2。",
            "- 数据披露：历史兼容脚本将 test 前 8 题同时作为 8-shot 示例和计分题；该结果是固定回归代理，不是无泄漏官方分数。", "",
        ]
        (run_dir / "e2e_summary.md").write_text("\n".join(md), encoding="utf-8")
        (run_dir / "e2e_accuracy_status.txt").write_text("PASS\n", encoding="utf-8")
    return summary


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_synthetic_run(root: Path) -> None:
    dataset = [{"question": f"Synthetic question {i}?", "answer": f"Work {i}. #### {10000+i}"} for i in range(GSM8K_TOTAL_ROWS)]
    dataset_path = root / "gsm8k_test.jsonl"
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in dataset), encoding="utf-8")
    prompts = expected_prompts(dataset)
    for name, latency in (("warmup", 2.0), ("timed", 1.0)):
        raw = [{"prompt_id": i, "prompt": prompts[i], "output": f" calculation #### {10000+i}\n", "correct": True} for i in range(NUM_QUESTIONS)]
        (root / f"gsm8k_raw_{name}.jsonl").write_text("\n".join(json.dumps(row) for row in raw), encoding="utf-8")
        result = {"task": "gsm8k", "backend": "srt", "num_gpus": 1, "latency": latency, "accuracy": 1.0, "num_requests": NUM_QUESTIONS, "other": {"num_questions": NUM_QUESTIONS, "parallel": PARALLEL}}
        (root / f"gsm8k_result_{name}.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
        (root / f"benchmark_{name}.log").write_text(f"Accuracy: 1.000\nInvalid: 0.000\nLatency: {latency:.3f} s\nOutput throughput: 10.000 token/s\n", encoding="utf-8")
    (root / "deepgemm_commit.txt").write_text(DG_COMMIT + "\n")
    (root / "sglang_commit.txt").write_text(SGLANG_COMMIT + "\n")
    (root / "sglang_source_sha256.txt").write_text(SGLANG_TAR_SHA256 + "\n")
    (root / "container_image_digest.txt").write_text(IMAGE_DIGEST + "\n")
    (root / "protocol_preregistered.snapshot.md").write_bytes(
        Path(__file__).with_name("protocol_preregistered.md").read_bytes()
    )
    write_json(root / "model_inventory.json", {"shard_count": MODEL_SHARDS, "shard_bytes": MODEL_BYTES})
    gpu_rows = ["index,name,uuid"] + [f"{i},NVIDIA H200,GPU-{i}" for i in range(8)]
    (root / "gpu_before.csv").write_text("\n".join(gpu_rows) + "\n")
    (root / "service_gpu_processes.csv").write_text("\n".join(f"GPU-{i},{100+i},100000" for i in range(8)) + "\n")
    (root / "service_server_processes.txt").write_text(
        "PID PPID USER STAT ELAPSED COMMAND\n"
        + "\n".join(f"{100+i} 1 root S 00:01 worker-{i}" for i in range(8))
        + "\n"
    )
    synthetic_extension_sha = "2" * 64
    identity = f"DEEPGEMM_COMMIT={DG_COMMIT}\nDEEPGEMM_PATH=/workspace/DeepGEMM/deep_gemm/__init__.py\nDEEPGEMM_EXTENSION_PATH=/workspace/DeepGEMM/deep_gemm/_C.synthetic.so\nDEEPGEMM_EXTENSION_SHA256={synthetic_extension_sha}\nSGLANG_PATH=/workspace/sglang/python/sglang/__init__.py\nTORCH_CUDA=13.0\nCUDA_DEVICE_COUNT=8\nGSM8K_HARNESS_SHA256=252f15aec6f1275eacf6bb58eaedb4198b3cf9c70af2f220d94495d0defb70e6\n"
    (root / "server_identity.log").write_text(identity)
    (root / "deepgemm_extension.sha256").write_text(
        f"{synthetic_extension_sha}  /tmp/DeepGEMM/deep_gemm/_C.synthetic.so\n"
    )
    build = "\n".join(
        f"[synthetic DP{rank} TP{rank} EP{rank}] Built sm90_nvfp4 MegaMoE weights for layer {layer} (block_n=256, grouped_nibbles=True)"
        for rank in range(8)
        for layer in range(3, 61)
    )
    actual_args = (
        "server_args=ServerArgs(quantization='modelopt_fp4', tp_size=8, dp_size=8, ep_size=8, "
        "enable_dp_attention=True, moe_a2a_backend='megamoe', disable_cuda_graph=True, "
        "disable_radix_cache=True, skip_server_warmup=True, max_running_requests=1316, "
        "watchdog_timeout=900.0, mem_fraction_static=0.9, enable_eplb=False)"
    )
    (root / "server.log").write_text(actual_args + "\n" + build + "\nThe server is fired up and ready to roll!\n")
    command = (
        "DG_SM90_NVFP4_BLOCK_N=256 DG_SM90_NVFP4_NIBBLE_GROUP=1 "
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096 "
        "DG_JIT_CACHE_DIR=/out/dg_jit_cache PYTHONPATH=/workspace/sglang/python:/workspace/DeepGEMM "
        f"python3 -m sglang.launch_server --model-path {MODEL_PATH} --quantization modelopt_fp4 "
        "--tp 8 --dp 8 --enable-dp-attention --moe-a2a-backend megamoe --disable-cuda-graph "
        "--disable-radix-cache --skip-server-warmup --max-running-requests 1316 "
        "--watchdog-timeout 900 --mem-fraction-static 0.9"
    )
    (root / "server_command.txt").write_text(command + "\n")
    write_json(
        root / "server_inspect.json",
        [{
            "Config": {
                "Image": IMAGE_REF,
                "Cmd": command.split()[5:],
                "Env": [
                    "DG_SM90_NVFP4_BLOCK_N=256",
                    "DG_SM90_NVFP4_NIBBLE_GROUP=1",
                    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096",
                    "DG_JIT_CACHE_DIR=/out/dg_jit_cache",
                    "PYTHONPATH=/workspace/sglang/python:/workspace/DeepGEMM",
                ],
            }
        }],
    )
    (root / "round_order.txt").write_text("warmup(excluded)->timed(formal)\n")
    (root / "retry_policy.txt").write_text("no_score_retry_or_replacement\n")
    grouped_jit = (
        "0" * 64 + "  ./cache/kernel.sm90_nvfp4_mega_moe_nibble_group.synthetic/kernel.cu\n"
        + "1" * 64 + "  ./cache/kernel.sm90_nvfp4_mega_moe_nibble_group.synthetic/kernel.cubin\n"
    )
    (root / "jit_cache_after_warmup.sha256").write_text(grouped_jit)
    (root / "jit_cache_after_timed.sha256").write_text(grouped_jit)
    (root / "grouped_nibble_jit_entries.txt").write_text(grouped_jit)


def self_test() -> None:
    assert MIN_CORRECT == 1260
    assert (MIN_CORRECT - 1) / NUM_QUESTIONS < MIN_REPORTED_ACCURACY
    assert MIN_CORRECT / NUM_QUESTIONS >= MIN_REPORTED_ACCURACY
    assert answer_value("x 1,234") == 1234
    assert answer_value("no number") == INVALID
    assert corruption_reasons("normal math #### 18") == []
    assert "repeated_fragment" in corruption_reasons("101" * 30)
    with tempfile.TemporaryDirectory(prefix="dsr1-e2e-selftest-") as tmp:
        root = Path(tmp)
        make_synthetic_run(root)
        service = validate_service_evidence(root)
        assert service["sm90_nvfp4_complete_rank_count"] == 8
        assert service["eplb_disabled_in_actual_server_args"] is True
        process_path = root / "service_gpu_processes.csv"
        process_text = process_path.read_text(encoding="utf-8")
        process_path.write_text("", encoding="utf-8")
        try:
            validate_service_evidence(root)
        except ValidationError as exc:
            assert "service_gpu_processes.csv is empty" in str(exc)
        else:
            raise AssertionError("empty service GPU process evidence was not rejected")
        process_path.write_text(process_text, encoding="utf-8")
        server_log_path = root / "server.log"
        server_log_text = server_log_path.read_text(encoding="utf-8")
        server_log_path.write_text(server_log_text.replace("enable_eplb=False", "enable_eplb=True"), encoding="utf-8")
        try:
            validate_service_evidence(root)
        except ValidationError as exc:
            assert "enable_eplb=False" in str(exc)
        else:
            raise AssertionError("enabled EPLB was not rejected")
        server_log_path.write_text(server_log_text, encoding="utf-8")
        summary = summarize_run(root, enforce_pinned_dataset=False, write_outputs=False)
        assert summary["status"] == "PASS"
        raw_path = root / "gsm8k_raw_timed.jsonl"
        rows = load_jsonl(raw_path)
        rows[17]["prompt"] += "MUTATED"
        raw_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        try:
            summarize_run(root, enforce_pinned_dataset=False, write_outputs=False)
        except ValidationError as exc:
            assert "prompt mismatch" in str(exc)
        else:
            raise AssertionError("prompt mutation was not rejected")
    print("DSR1_E2E_PARSER_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-dataset", type=Path)
    parser.add_argument("--validate-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-round", choices=("warmup", "timed"))
    parser.add_argument("--validate-service-evidence", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        elif args.validate_dataset:
            value = validate_dataset(args.validate_dataset)
            require(args.output is not None, "--output is required")
            write_json(args.output, value)
        elif args.validate_model:
            value = validate_model(args.validate_model)
            require(args.output is not None, "--output is required")
            write_json(args.output, value)
        elif args.validate_service_evidence:
            require(args.run_dir is not None, "--run-dir is required")
            value = validate_service_evidence(args.run_dir)
            if args.output:
                write_json(args.output, value)
            else:
                print(json.dumps(value, sort_keys=True))
        elif args.validate_round:
            require(args.run_dir is not None, "--run-dir is required")
            dataset = load_jsonl(args.run_dir / "gsm8k_test.jsonl")
            value = asdict(validate_round(args.run_dir, args.validate_round, dataset))
            if args.output:
                write_json(args.output, value)
            else:
                print(json.dumps(value, sort_keys=True))
        elif args.run_dir:
            print(json.dumps(summarize_run(args.run_dir), sort_keys=True))
        else:
            parser.error("select one operation")
    except (ValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"VALIDATION_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
