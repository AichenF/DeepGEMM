# Iteration 716a: compose a unique 64-register-capped RDC candidate

## Hypothesis

The natural-register unique RDC candidate recovered standalone W2's 32-QGMMA / 16-DEPBAR schedule but allocated 195 registers and a 112-byte stack per thread. Separately, the standalone W2 control retained 32/16 scheduling under a 64-register/eight-CTA contract and serialized only when constrained to 56 registers/nine CTAs. The untested composition is therefore to cap the RDC translation unit at 64 registers, aligning caller and callee contracts while preserving the phase-local compilation boundary.

## Change and gates

`bench/iter716_make_unique_rdc_maxrreg.py` copies the exact unique Iteration-714 RDC inputs, assigns fresh host and CUDA entry symbols, and adds only `--maxrregcount=64` to the CUDA compile flags. It does not alter production source, math, task order, barriers, communication, or Python ABI.

The candidate is rejected before launch unless all of the following hold:

- device compile/link succeeds without caller/callee register-contract errors;
- TP4 M128 split-K2 entry has at most 64 registers and no fixed local allocation or material stack/spill cliff;
- the exact W2 callee retains approximately 32 QGMMAs / 16 dependency barriers.

This is a composition checkpoint only. No JIT, CUDA business-kernel launch, correctness, or timing is claimed before commit.
