# Iteration 715b: repair already-unique host-symbol composition

The first invocation of the committed Iteration-715a composer stopped before NVCC because it expected four occurrences of the tagged host function in the already-unique Iteration-714 `main.cpp`, but that source correctly contains only two: the C++ declaration and the pybind function reference. The Python-visible method name and doc string had already been restored to their canonical strings by the prior composer and therefore do not contain the old tag.

The repair changes the exact-count assertion from four to two and removes the now-inapplicable second string rewrite. CUDA source, build flags, production source, math, scheduling and ABI are untouched. The partial `/tmp/iter715a_unique_dlto` directory is retained as failure evidence; the next attempt will use a fresh `/tmp/iter715b_unique_dlto` directory.

No NVCC compile, device link, CUDA business-kernel launch, correctness test or timing occurred in the failed attempt.
