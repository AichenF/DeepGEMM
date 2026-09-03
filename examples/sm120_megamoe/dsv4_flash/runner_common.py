"""Shared validation and topology helpers for SM120 MegaMoE runners."""

from __future__ import annotations

import hashlib
from pathlib import Path

IMAGE = "nvcr.io/nvidia/pytorch:26.07-py3"

# GPU index -> InfiniBand HCA on the same PCIe switch on R6KD-CX8.
GPU_HCA = {
    0: "mlx5_2",
    1: "mlx5_3",
    2: "mlx5_0",
    3: "mlx5_1",
    4: "mlx5_6",
    5: "mlx5_7",
    6: "mlx5_4",
    7: "mlx5_5",
}
GPU_UVERBS = {0: 2, 1: 3, 2: 0, 3: 1, 4: 6, 5: 7, 6: 4, 7: 5}
NUMA_CPUS = {0: "0-55,112-167", 1: "56-111,168-223"}

DEFAULT_CONFIG = {
    "CAKE_MOE_HIDDEN": 4096,
    "CAKE_MOE_INTERMEDIATE": 4096,
    "CAKE_MOE_OUTPUT": 4096,
    "CAKE_MOE_TOPK": 6,
    "CAKE_MOE_LOCAL_EXPERTS": 32,
    "CAKE_MOE_PHYSICAL_RANKS": 8,
    "CAKE_MOE_MAX_ROWS": 2048,
}


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_gpus(value: str, world_size: int) -> list[int]:
    """Parse and validate an exact rank-to-physical-GPU assignment."""
    try:
        gpus = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"invalid GPU list {value!r}") from exc
    if len(gpus) != world_size:
        raise ValueError(
            f"world size {world_size} requires exactly {world_size} GPUs, got {gpus}"
        )
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU list contains duplicates: {gpus}")
    unknown = [gpu for gpu in gpus if gpu not in GPU_HCA]
    if unknown:
        raise ValueError(f"GPU indices have no topology mapping: {unknown}")
    return gpus


def topology(gpus: list[int]) -> dict[str, str | list[str]]:
    """Resolve the devices and NUMA placement for a physical GPU list."""
    numas = sorted({0 if gpu < 4 else 1 for gpu in gpus})
    devices = ["--device=/dev/infiniband/rdma_cm"]
    devices.extend(f"--device=/dev/infiniband/uverbs{GPU_UVERBS[gpu]}" for gpu in gpus)
    return {
        "devices": devices,
        "cpuset": ",".join(NUMA_CPUS[numa] for numa in numas),
        "memset": ",".join(str(numa) for numa in numas),
        "hcas": ",".join(GPU_HCA[gpu] for gpu in gpus),
    }


def read_build_config(build: Path) -> tuple[list[str], dict[str, int]]:
    """Read the compiler definitions that identify a build directory."""
    config_path = build / "config-flags.txt"
    if not config_path.is_file():
        raise ValueError(f"missing config-flags.txt in {build}")
    flags = config_path.read_text().split()
    config = dict(DEFAULT_CONFIG)
    for flag in flags:
        if not flag.startswith("-D") or "=" not in flag:
            continue
        name, value = flag[2:].split("=", 1)
        if name in config:
            try:
                config[name] = int(value, 0)
            except ValueError as exc:
                raise ValueError(f"non-integer shape definition {flag!r}") from exc
    if "CAKE_MOE_OUTPUT" not in {
        flag[2:].split("=", 1)[0]
        for flag in flags
        if flag.startswith("-D") and "=" in flag
    }:
        config["CAKE_MOE_OUTPUT"] = config["CAKE_MOE_HIDDEN"]
    return flags, config


def validate_build(
    build: Path,
    *,
    binary: str,
    world_size: int,
    rows: int,
    hidden: int,
    intermediate: int,
    experts: int,
    topk: int,
) -> dict:
    """Fail closed when a binary was compiled for a different workload."""
    artifact = build / binary
    if not artifact.is_file():
        raise ValueError(f"missing {binary} binary in {build}")
    flags, config = read_build_config(build)
    expected = {
        "CAKE_MOE_HIDDEN": hidden,
        "CAKE_MOE_INTERMEDIATE": intermediate,
        "CAKE_MOE_OUTPUT": hidden,
        "CAKE_MOE_TOPK": topk,
    }
    mismatches = [
        f"{name}={config[name]} (expected {value})"
        for name, value in expected.items()
        if config[name] != value
    ]
    if config["CAKE_MOE_LOCAL_EXPERTS"] * world_size != experts:
        mismatches.append(
            "CAKE_MOE_LOCAL_EXPERTS="
            f"{config['CAKE_MOE_LOCAL_EXPERTS']} gives "
            f"{config['CAKE_MOE_LOCAL_EXPERTS'] * world_size} experts "
            f"at world size {world_size} (expected {experts})"
        )
    if config["CAKE_MOE_PHYSICAL_RANKS"] < world_size:
        mismatches.append(
            f"CAKE_MOE_PHYSICAL_RANKS={config['CAKE_MOE_PHYSICAL_RANKS']} "
            f"is smaller than world size {world_size}"
        )
    if config["CAKE_MOE_MAX_ROWS"] < rows:
        mismatches.append(
            f"CAKE_MOE_MAX_ROWS={config['CAKE_MOE_MAX_ROWS']} is smaller than {rows}"
        )
    if mismatches:
        raise ValueError(
            f"build {build} does not match the workload: " + "; ".join(mismatches)
        )
    return {
        "path": str(build.resolve()),
        "binary": binary,
        "binary_sha256": file_sha256(artifact),
        "config_sha256": file_sha256(build / "config-flags.txt"),
        "config_flags": flags,
        "config": config,
    }
