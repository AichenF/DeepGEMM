# SM90 NVFP4 Braided LUT-Offset Integration

## Goal

Reduce the exposed Pro fused-kernel decode cost without changing the standard
NVFP4 values, persistent weight size, TMA payload, shared-memory footprint, or
runtime API.

## Representation

The retained model-load prepack uses an 80-byte BK128 row: 64 bytes of braided
E2M1 values, eight UE4M3 scale bytes, and eight padding bytes. The candidate
uses the same 80 bytes and rewrites the metadata region as eight little-endian
`uint16` values equal to `scale_code * sizeof(uint2)`. They are direct byte
offsets into the existing 128-entry shared LUT. The canonical scale tensor is
kept unchanged, so the cached representation remains losslessly derived from
standard E2M1 plus one UE4M3 scale per 16 values.

The follow-up revision also pre-applies the WGMMA shared-memory row swizzle.
For row `r`, destination K16 group `h` stores original group
`h ^ (r & 7)`, and the scale offsets follow the same permutation. The decoder
then writes groups linearly. Its final 128 decoded bytes are identical to the
retained decoder's `group_offset ^ ((r & 7) << 4)` stores; only the deployment
order and address arithmetic differ.

The decoder replaces one 64-bit scale load plus eight mask/shift/address-scale
chains with one 128-bit metadata load and direct LUT byte addresses, then
removes the runtime XOR from every decoded output-group store. FP8 weights or
FP8 lookup rows are never stored in the deployment cache.

## Isolation And Gate

The candidate has separate API, JIT runtime, CUDA header, Python prepack, and
benchmark adapters. It reuses the proxy-safe braided three-stage fused body;
no production wrapper, environment variable, or runtime argument changes.

Advance only if fresh JIT shows exact-NVFP4 correctness for both global-scale
modes, 168 registers, a 56-byte stack, zero spills, and SASS removes the eight
scale-address shifts without adding local traffic. Then require repeated
multi-seed ABBA improvement at M16 and M64 against the identical proxy-safe
three-stage baseline. Remove the candidate if the endpoint result is neutral
or negative.
