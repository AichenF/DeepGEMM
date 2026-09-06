# Iteration 724b — CUDA 12.8 emits a real terminal device boundary

## Result

The compiler-mechanics gate passes on the actual container toolchain:

```text
CUDA compilation tools, release 12.8, V12.8.61
```

For the matched probe in
`bench/probes/iter724_noreturn_device_probe.cu`, NVCC generates:

```text
.func _Z13terminal_tailPii(
...
.noreturn
```

The terminal SASS control flow is:

```text
terminal_entry:
    CALL.REL.NOINC <terminal callee>
terminal callee:
    ...
    EXIT
```

There is no terminal-callee `RET` and no reachable caller continuation.  The
matched returning entry instead has a caller-side `EXIT` after its call and
the returning callee ends in `RET.REL.NODEC`.

`ptxas -v` reports zero stack frame and zero spill stores/loads for both
entries and both callees.  The terminal entry uses eight registers versus ten
for the returning entry in this small probe.

## Interpretation and next gate

This proves only that CUDA 12.8 preserves a materially distinct no-return ABI
through PTX and SASS.  It does not prove that the large W2 body will regain
standalone scheduling, and no business kernel was launched.

Proceed with a default-off, M128-only compile/SASS probe that calls a
`[[noreturn]]` whole-W2 callee after activation.  The callee may temporarily
terminate immediately after W2 because runtime execution is forbidden at
this stage.  Admit collective migration only if exact W2 SASS retains all 32
QGMMAs and materially reduces the current 32 dependency waits toward the
standalone count of 16 without a prohibitive register/stack cliff.

## Artifacts

- `bench/probes/iter724_noreturn_device_probe.cu`
- `bench/results/iter724b_noreturn_device_probe_compile_20260906.log`
- `bench/results/iter724b_noreturn_device_probe_20260906.ptx`
- `bench/results/iter724b_noreturn_device_probe_20260906.sass`
- `bench/results/iter724b_noreturn_device_probe_20260906.cubin`
