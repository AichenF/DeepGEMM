# Iteration 713s: selectively dispatch only the single kernel to RDC

Date: 2026-09-06

The opt-in prebuilt-extension runner can now load two shared libraries with
the same CPython extension name from distinct paths.  With
`V4_PREBUILT_CONTROL_EXTENSION` and
`V4_PREBUILT_CANDIDATE_SYMBOLS` set, it returns a method-level proxy to the
intercepted `load_inline` request:

- listed candidate symbols dispatch to the RDC library;
- every other method dispatches to the ordinary non-RDC control library.

The first probe selects only `run_tp4_megamoe_single_launch`.  Route alignment,
standalone W13/W2, SwiGLU/requant, k6 reduction, the multi-kernel graph and all
other extension methods remain owned by the already correctness-passing
ordinary library.  This prevents Iteration 713q's invalid RDC multi control
from contaminating the candidate workspace and directly tests whether the RDC
single business kernel is numerically valid.

The runner validates identical module filenames, existence of every selected
symbol in both libraries, and prints exact dispatch provenance on rank zero.
The mode is default-off and changes neither library nor benchmark source.
Python compilation passes; no CUDA launch is claimed for this composition.
