#!/usr/bin/env python3
"""Run a Python benchmark while substituting one prebuilt JIT extension."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import runpy
import sys
import types

import torch.utils.cpp_extension as cpp_extension


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: iter713_prebuilt_extension_runner.py TARGET [ARGS...]")

    extension_path = Path(os.environ["V4_PREBUILT_EXTENSION"]).resolve()
    if not extension_path.is_file():
        raise FileNotFoundError(extension_path)
    extension_name = extension_path.name.removesuffix(".so")

    spec = importlib.util.spec_from_file_location(extension_name, extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension spec for {extension_path}")
    extension = importlib.util.module_from_spec(spec)
    sys.modules[extension_name] = extension
    spec.loader.exec_module(extension)

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

        def install_namespace(package_name: str, package_dir: Path) -> None:
            if not package_dir.is_dir():
                raise FileNotFoundError(package_dir)
            package = types.ModuleType(package_name)
            package.__package__ = package_name
            package.__path__ = [str(package_dir)]
            sys.modules[package_name] = package

        install_namespace("sglang", sglang_package_dir)
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

    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
