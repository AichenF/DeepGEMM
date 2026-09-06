# Iteration 713u: non-RDC rebuild command quoting failure

Date: 2026-09-06

The first same-ABI non-RDC rebuild command exits in the Python `-c` parser.
The remote shell passed literal `\\x27` sequences inside a dictionary
expression, producing `SyntaxError: unexpected character after line
continuation character`.

No module import, JIT compilation, CUDA context, or kernel launch occurred.
Retry with a plain positional `print()` expression that requires no nested
quote escaping.

Raw log: `bench/results/iter713u_miniforge_nonrdc_build_20260906.log`.
