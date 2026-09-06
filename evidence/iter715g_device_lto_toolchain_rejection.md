# Iteration 715g: reject device LTO on the available CUDA 12.8 toolchain

## Static result

The final single-image composition compiled the generated CUDA source into an LTO object, but device link still emitted architecture-generic PTX. An explicit `nvcc ... -dlink -dlto -arch=sm_90a -ptx` diagnostic produced:

```text
// Cuda compilation tools, release 12.8, V12.8.61
.version 8.7
.target sm_90
.address_size 64
...
wgmma.fence.sync.aligned;
```

The ordinary device link therefore failed deterministically when ptxas rejected `wgmma.fence`, FP8 `wgmma.mma_async`, `wgmma.commit_group`, and `wgmma.wait_group` on `.target sm_90`. The container exposes only `/usr/local/cuda-12.8`; there is no newer NVCC available for an equivalent check.

Artifact hashes:

- CUDA LTO object: `f58b533a8a721e93e33b81691f144fe1be76709ad4bd67ab6a24c65f64fbb40b`
- nvlink-generated PTX: `d756d0cfb3b0310ec6abaaf36ce0edfcb24918f1b1e4ec812e1956e28cb98269`
- generated build recipe: `71207b880eab6a68048ee4159eb8bab7ee841f3df7c28272c16b05383ca25def`

## Decision

Reject device LTO on this benchmark's available toolchain. It cannot reach the resource/SASS gate because it cannot emit a valid architecture-specific WGMMA cubin. No CUDA business kernel was launched, so there is no correctness or latency number to compare. Production kernel and benchmark sources remain unchanged.

This closes the remaining compiler-boundary transfer from multi-kernel: ordinary same-TU outline, natural register allocation, function attributes, RDC device calls, unique-symbol attribution, and now device LTO have all failed either static or runtime gates. Further work must change the one-kernel W13/W2 dataflow rather than attempt another equivalent compilation boundary.
