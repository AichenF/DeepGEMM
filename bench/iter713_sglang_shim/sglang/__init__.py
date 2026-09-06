"""Select a source checkout without importing SGLang's frontend APIs."""

from __future__ import annotations

import os
from pathlib import Path


_package_dirs = [
    Path(value.strip()).resolve() / "sglang"
    for value in os.environ.get(
        "V4_SGLANG_NAMESPACE_OVERLAY_ROOTS", ""
    ).split(",")
    if value.strip()
]
_package_dirs.append(
    Path(os.environ["V4_SGLANG_NAMESPACE_ROOT"]).resolve() / "sglang"
)
for _package_dir in _package_dirs:
    if not _package_dir.is_dir():
        raise FileNotFoundError(_package_dir)
__path__ = [str(_package_dir) for _package_dir in _package_dirs]
