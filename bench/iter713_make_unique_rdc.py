#!/usr/bin/env python3
"""Create a uniquely named copy of the temporary Iteration-713 RDC build."""

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
    parser.add_argument("--tag", default="iter714a")
    args = parser.parse_args()

    if not args.tag.replace("_", "").isalnum():
        raise ValueError("--tag must be alphanumeric/underscore")
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

    unique_host = f"{HOST_ENTRY}_{args.tag}"
    unique_cuda = f"{CUDA_ENTRY}_{args.tag}"
    cuda_source = output_dir / "cuda.cu"
    host_source = output_dir / "main.cpp"
    build_file = output_dir / "build.ninja"

    rewrite(cuda_source, CUDA_ENTRY, unique_cuda, 5)
    rewrite(cuda_source, HOST_ENTRY, unique_host, 1)
    rewrite(host_source, HOST_ENTRY, unique_host, 4)
    # Keep the Python-visible method and its doc string canonical while the
    # C++ declaration/reference remain uniquely named.
    rewrite(host_source, f'"{unique_host}"', f'"{HOST_ENTRY}"', 2)
    rewrite(build_file, str(source_dir), str(output_dir), 2)

    print(
        json.dumps(
            {
                "cuda_entry": unique_cuda,
                "cuda_sha256": sha256(cuda_source),
                "host_entry": unique_host,
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
