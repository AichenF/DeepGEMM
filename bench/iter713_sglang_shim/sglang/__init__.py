"""Select a source checkout without importing SGLang's frontend APIs."""

from __future__ import annotations

import os
from pathlib import Path


_package_dir = Path(os.environ["V4_SGLANG_NAMESPACE_ROOT"]).resolve() / "sglang"
if not _package_dir.is_dir():
    raise FileNotFoundError(_package_dir)
__path__ = [str(_package_dir)]
