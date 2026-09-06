# Iteration 713m: IPC header compatibility failure

Date: 2026-09-06

With the disk namespace shim active, CARv2's P2P capability subprocess
completes and all ranks advance to `IPCManager` initialization.  The original
old-checkout `distributed/ipc.cuh` then fails under the fixed TVM-FFI 0.1.11
headers: free `get<0>(pair)` and `get<1>(pair)` cannot resolve for the const
`tvm::ffi::Tuple` reference.

This occurs before communicator construction, CUDA Graph capture,
correctness, timing, or any MegaMoE business-kernel launch.

The already established baseline compatibility overlay still exists at:

```text
/home/xutingz/fac/.tpmoe_tmp/sglang/jit_kernel
```

It is a copy of the checkout JIT-kernel tree and changes only the two IPC tuple
accesses:

```diff
- get<0>(pair)
- get<1>(pair)
+ pair.template get<0>()
+ pair.template get<1>()
```

The patched `ipc.cuh` SHA256 is
`ba6fec5228a4c349ae3eb3e02f2494b1c309f8c2fa9014c08b079fc1089424fc`,
and the corresponding successful TVM-FFI cache module from prior accepted
baselines remains present.

Decision: extend the namespace package search path with the overlay first and
the original checkout second.  This preserves the checkout and reuses the
same compatibility path as the frozen baseline.
