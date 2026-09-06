#!/usr/bin/env python3
"""Run a Python benchmark while substituting one prebuilt JIT extension."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import types

import torch.utils.cpp_extension as cpp_extension


def load_extension(extension_path: Path, extension_name: str) -> object:
    spec = importlib.util.spec_from_file_location(extension_name, extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension spec for {extension_path}")
    extension = importlib.util.module_from_spec(spec)
    sys.modules[extension_name] = extension
    spec.loader.exec_module(extension)
    return extension


class SelectiveExtension:
    def __init__(
        self,
        control: object,
        candidate: object,
        candidate_symbols: frozenset[str],
    ) -> None:
        self._control = control
        self._candidate = candidate
        self._candidate_symbols = candidate_symbols

    def __getattr__(self, name: str) -> object:
        owner = (
            self._candidate if name in self._candidate_symbols else self._control
        )
        return getattr(owner, name)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: iter713_prebuilt_extension_runner.py TARGET [ARGS...]")

    extension_path = Path(os.environ["V4_PREBUILT_EXTENSION"]).resolve()
    if not extension_path.is_file():
        raise FileNotFoundError(extension_path)
    extension_name = extension_path.name.removesuffix(".so")

    control_path_value = os.environ.get("V4_PREBUILT_CONTROL_EXTENSION")
    if control_path_value:
        control_path = Path(control_path_value).resolve()
        if not control_path.is_file():
            raise FileNotFoundError(control_path)
        if control_path.name.removesuffix(".so") != extension_name:
            raise ValueError(
                "candidate and control extension filenames must have the same "
                "module name"
            )
        candidate_symbols = frozenset(
            value.strip()
            for value in os.environ.get(
                "V4_PREBUILT_CANDIDATE_SYMBOLS", ""
            ).split(",")
            if value.strip()
        )
        if not candidate_symbols:
            raise ValueError(
                "V4_PREBUILT_CANDIDATE_SYMBOLS is required with a control "
                "extension"
            )
        control_extension = load_extension(control_path, extension_name)
        # Both libraries export the same CPython module initializer.  Load
        # each from its distinct path, retain both objects, and expose a
        # method-level dispatcher only to the intercepted load_inline call.
        sys.modules.pop(extension_name, None)
        candidate_extension = load_extension(extension_path, extension_name)
        for symbol in candidate_symbols:
            getattr(control_extension, symbol)
            getattr(candidate_extension, symbol)
        extension = SelectiveExtension(
            control_extension, candidate_extension, candidate_symbols
        )
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "ITER713_PREBUILT_DISPATCH "
                + json.dumps(
                    {
                        "candidate": str(extension_path),
                        "candidate_symbols": sorted(candidate_symbols),
                        "control": str(control_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        extension = load_extension(extension_path, extension_name)

    original_load_inline = cpp_extension.load_inline

    def load_inline(name: str, *args: object, **kwargs: object) -> object:
        if name == extension_name:
            return extension
        return original_load_inline(name, *args, **kwargs)

    cpp_extension.load_inline = load_inline

    sglang_root_value = os.environ.get("V4_SGLANG_NAMESPACE_ROOT")
    if sglang_root_value:
        sglang_package_dir = Path(sglang_root_value).resolve() / "sglang"
        if not sglang_package_dir.is_dir():
            raise FileNotFoundError(sglang_package_dir)

        def install_namespace(
            package_name: str, package_dirs: Path | list[Path]
        ) -> None:
            if isinstance(package_dirs, Path):
                package_dirs = [package_dirs]
            for package_dir in package_dirs:
                if not package_dir.is_dir():
                    raise FileNotFoundError(package_dir)
            package = types.ModuleType(package_name)
            package.__package__ = package_name
            package.__path__ = [str(package_dir) for package_dir in package_dirs]
            sys.modules[package_name] = package

        overlay_roots = os.environ.get(
            "V4_SGLANG_NAMESPACE_OVERLAY_ROOTS", ""
        ).split(",")
        overlay_dirs = [
            Path(value.strip()).resolve() / "sglang"
            for value in overlay_roots
            if value.strip()
        ]
        install_namespace("sglang", overlay_dirs + [sglang_package_dir])
        subpackages = os.environ.get(
            "V4_SGLANG_NAMESPACE_SUBPACKAGES", ""
        ).split(",")
        for package_name in (value.strip() for value in subpackages):
            if not package_name:
                continue
            if not package_name.startswith("sglang."):
                raise ValueError(
                    "V4_SGLANG_NAMESPACE_SUBPACKAGES entries must start "
                    "with 'sglang.'"
                )
            relative = package_name.removeprefix("sglang.").replace(".", "/")
            install_namespace(package_name, sglang_package_dir / relative)

        if os.environ.get("V4_SGLANG_LIGHT_MOE_ALIGN", "0") == "1":
            # The compatibility overlay intentionally owns jit_kernel, but
            # may also contain namespace-only kernels stubs.  Load the real
            # registry-backed MoE op package from the selected checkout.
            install_namespace(
                "sglang.kernels", sglang_package_dir / "kernels"
            )
            install_namespace(
                "sglang.kernels.ops", sglang_package_dir / "kernels" / "ops"
            )
            for package_name in (
                "sglang.srt.layers.moe",
                "sglang.srt.layers.moe.moe_runner",
                "sglang.srt.layers.moe.moe_runner.triton_utils",
            ):
                relative = package_name.removeprefix("sglang.").replace(
                    ".", "/"
                )
                install_namespace(package_name, sglang_package_dir / relative)
            leaf = importlib.import_module(
                "sglang.srt.layers.moe.moe_runner.triton_utils."
                "moe_align_block_size"
            )
            public_name = "sglang.srt.layers.moe.fused_moe_triton"
            public_module = types.ModuleType(public_name)
            public_module.__package__ = public_name
            public_module.moe_align_block_size = leaf.moe_align_block_size
            sys.modules[public_name] = public_module

    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
