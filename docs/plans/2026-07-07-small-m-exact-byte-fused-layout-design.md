# Small-M exact-byte packed-NVFP4 tile layout

## Context

Iteration 73 reduced the official `128x4096x7168/E8` latency to
`252.830 us`, or `11.016%` bound-aware roofline SOL.  Its generated cubin uses
80 registers per thread, no local memory, and about 25 KiB dynamic shared
memory, allowing three 256-thread CTAs per H20 SM.  Raising occupancy alone to
four CTAs cannot close the remaining 6.35x gap.

SASS and source inspection show that the producer repeatedly computes two
independent addresses and issues separate transactions for packed FP4 and its
UE4M3 scale.  The `megamoe_nvfp4` reference branch uses a co-located packed
weight/scale representation to reduce this loader work, but its 80-byte row
stride includes padding.  Padding is not acceptable here because the user
requires true packed FP4 without increased persistent weight bytes.

## Chosen design

For `M <= 128` only, `prepare()` replaces the two pretransformed tensors with
one exact-byte uint8 buffer.  Every K128 tile contains two aligned,
vector-major sections:

```
packed section: 4 x [N rows, 16 bytes]
scale section:  2 x [N rows,  4 bytes]
```

The physical size is therefore exactly

```
E * N * (K / 2 + K / 16)
```

bytes, identical to the canonical packed-E2M1 payload plus its existing
UE4M3 block scales.  No padding, decoded FP8 copy, residual component, or
duplicate scale tensor is retained.  The existing per-expert FP32 global
scale remains separate.  A zero-element FP8 tensor occupies the unchanged
`block_scale` API argument; the kernel never dereferences it for this layout.

The fused tensor uses a distinct four-dimensional marker shape so the C++ API
can distinguish it from the existing `[E, K/16, N, 8]` pretransform.  The
public Python function signature and canonical model input remain unchanged.
For `M > 128`, `prepare()` and every large-M kernel keep the existing layout
and dispatch byte-for-byte.

## Producer data flow

The small-M producer maps 128 threads to 64 output rows and two K64 halves.
For each K128 tile, each thread obtains:

- two aligned and warp-coalesced 128-bit loads containing 32 packed FP4 bytes;
- one aligned 32-bit load containing four original UE4M3 scale bytes.

It then performs the existing on-chip E2M1-times-UE4M3 lookup and writes only
the transient FP8 WGMMA stage to shared memory.  Arithmetic, output scaling,
WGMMA orientation, and expert routing remain unchanged.

## Alternatives

1. Forcing 64 registers can raise the theoretical resident-CTA count from
   three to four, but offers at most a 33% concurrency increase and may spill.
2. TMA-loading the current disjoint layout can hide some memory latency but
   retains loader address work and integer decode; it also adds descriptors,
   shared memory, and barriers.
3. The reference branch's padded 80-byte row format improves alignment but
   increases persistent bytes by 11.1%, violating the storage constraint.

## Validation gates

1. Prepared packed payload plus scale payload has exactly the same number of
   bytes as the canonical tensors for every tested shape; global scale bytes
   are unchanged.
2. The prepared payload remains uint8 packed E2M1 plus the original bytewise
   E4M3 scales.  No persistent decoded FP8 weight exists.
3. Cross-pass and partial-expert probes have zero Cupra elementwise
   violations and no NaN/Inf.
4. Iteration timing uses eight slots, ten warmups, thirty cold-L2 trials.
5. The existing large-M `8192x4096x7168/E8` path is source- and dispatch-
   unchanged until final regression validation.
