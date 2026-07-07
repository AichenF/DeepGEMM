# SM90 Dense NVFP4 Single-BF16 Staging Fallback

## Decision

Iteration 56 proves that one E4M3 staging term is fast but cannot satisfy the unchanged Cupra elementwise contract for either K=7168 shape. The exact fallback will therefore decode each packed NVFP4 weight once into one transient BF16 shared-memory value and execute one logical BF16 WGMMA term. It will not restore primary/residual FP8 decomposition.

## Representation and data flow

- Global memory remains packed E2M1 codes, UE4M3 block scales, and one FP32 global scale per expert.
- `prepare()` remains a dtype- and numel-preserving permutation.
- The producer converts FP8 activations to BF16 exactly and expands E2M1 times UE4M3 to BF16 exactly in shared memory.
- The consumer uses Hopper BF16 WGMMA with FP32 accumulation. The epilogue applies only activation and global scales; the previous FP8 `/8` normalization compensation is removed.
- There is one B staging tile and one WGMMA stream, not primary and residual tiles or streams.

All E4M3 activation values are exactly representable in BF16. E2M1 times E4M3 has at most the combined source significand precision and is also exactly representable in BF16 over the task's range.

## Layout and resources

The existing two-stage producer/consumer pipeline and persistent tile scheduler remain. A and B stage elements grow from one to two bytes. The dense lookup table grows from 1 KiB of E4M3 entries to 2 KiB of BF16 entries. BLOCK_N=256 still fits within Hopper's dynamic shared-memory capacity.

The BF16 WGMMA K atom is 16 rather than FP8's 32, so each K=128 stage issues eight BF16 WGMMA instructions. This is expected to trade some of Iteration 56's speed for exactness while avoiding the semantic and storage complexity of two FP8 weight components.

## Acceptance

1. Rebuild in the Python 3.12 `lmsysorg/sglang:dev` container with a fresh JIT cache.
2. Run all six fixed Cupra shapes without changing the checker.
3. Require 6/6 PASS and zero elementwise violations on both K=7168 shapes.
4. Report useful FLOPs only and compare latency against both single-E4M3 Iteration 56 and primary/residual Iteration 50.
5. Reconfirm prepared dtype, numel, and total persistent bytes are unchanged.
