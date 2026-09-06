# Iteration 714a: compose a uniquely named RDC single entry

Date: 2026-09-06

Add `bench/iter713_make_unique_rdc.py`, a count-checked mechanical composer for
the already built Iteration-713f generated-source probe.  It copies only
`cuda.cu`, `main.cpp`, and `build.ninja` into a fresh requested directory and:

- uniquely renames the TP4 CUDA entry template and every launch reference;
- uniquely renames the C++ host entry declaration/definition/reference;
- keeps the Python-visible pybind method name
  `run_tp4_megamoe_single_launch` unchanged;
- retargets the two absolute Ninja source paths to the new directory.

Every replacement has an asserted occurrence count, the output directory must
not already exist, and the script reports exact source/build hashes.  All math,
launch arguments, RDC flags, device callees and other generated code remain
byte-identical.  Unique ELF and CUDA symbols remove the remaining dynamic
symbol-preemption ambiguity from Iterations 713x/713y.

The script passes `py_compile`.  This commit records the composition before it
is applied or compiled; no CUDA work is claimed.
