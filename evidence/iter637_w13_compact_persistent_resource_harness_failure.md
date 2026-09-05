# Iteration 637: compact-W13 persistent-state resource harness failure

The new default-off candidate keeps the selected compact W13 phase outline
and enables persistent route-GEMM state only inside its M128 grid-stride loop.
This is distinct from the historical all-GEMM persistent switch, which also
changed W2 and disabled the compact outline.

The SM90a import completed and printed the new extension identity:

```text
v4tp_7babc76b617baae01065_v178mspec
```

However, the wrapper then selected an unordered recent-file result rather
than constructing the path from that identity:

```text
EXT=/tmp/torch_ext_v4_tp/v4tp_f72712dd95f6ccb3ff4e_v178mspec/
    v4tp_f72712dd95f6ccb3ff4e_v178mspec.so
```

That artifact is Iteration 634's W2-persistent cubin.  Its resource rows are
valid for that earlier candidate but provide no evidence for compact-W13
persistent state.  No CUDA kernel was launched.  The resource gate must be
repeated against the exact
`v4tp_7babc76b617baae01065_v178mspec.so` artifact before correctness or timing.
