#!/usr/bin/env python3
"""Create a uniquely named device-LTO copy of an Iteration-714 RDC build."""

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
    parser.add_argument("--source-tag", default="iter714b")
    parser.add_argument("--tag", default="iter715a")
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
    cuda_source = output_dir / "cuda.cu"
    host_source = output_dir / "main.cpp"
    build_file = output_dir / "build.ninja"

    rewrite(cuda_source, old_cuda, new_cuda, 5)
    rewrite(cuda_source, old_host, new_host, 1)
    # The Iteration-714 input is already unique: only the C++ declaration and
    # function reference carry its tag, while the Python method/doc string are
    # canonical.  Rename exactly those two C++ occurrences.
    rewrite(host_source, old_host, new_host, 2)
    rewrite(build_file, str(source_dir), str(output_dir), 2)

    # Emit LTO IR at compile time, then enable the optimization itself at
    # device link.  With explicit gencode targets NVCC rejects a compile-side
    # -dlto flag; code=lto_90a is the compile-side request.
    ninja_text = build_file.read_text()
    ninja_lines = ninja_text.splitlines(keepends=True)
    cuda_flag_lines = [
        index
        for index, line in enumerate(ninja_lines)
        if line.startswith("cuda_cflags = ")
    ]
    if len(cuda_flag_lines) != 1:
        raise RuntimeError(
            f"{build_file}: expected one cuda_cflags line, "
            f"got {len(cuda_flag_lines)}"
        )
    cuda_flag_index = cuda_flag_lines[0]
    cuda_flag_line = ninja_lines[cuda_flag_index]
    if cuda_flag_line.count("code=sm_90a") != 2:
        raise RuntimeError(
            f"{build_file}: expected two sm_90a compile targets"
        )
    ninja_lines[cuda_flag_index] = cuda_flag_line.replace(
        "code=sm_90a", "code=lto_90a"
    )
    build_file.write_text("".join(ninja_lines))
    rewrite(build_file, " -dlink -gencode=", " -dlink -dlto -gencode=", 1)

    print(
        json.dumps(
            {
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
