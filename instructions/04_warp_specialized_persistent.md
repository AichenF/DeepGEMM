## W2 warp-specialized persistent direction

Sources consulted:

- `cudac-tensor-core-expert/references/group_gemm_01_optimizations.md`
- `cudac-tensor-core-expert/references/optimization_02_persistent_kernel.md`
- `cudac-tensor-core-expert/references/optimization_04_warp_specialization.md`
- `cudac-tensor-core-expert/references/optimization_01_mbarrier.md`
- `cudac-tensor-core-expert/references/wgmma_04_tma_loading_pattern.md`
- `cudac-tensor-core-expert/references/wgmma_05_execution.md`

Implementation constraints distilled for this kernel:

1. Launch a fixed, occupancy-sized worker grid and let each CTA iterate a
   deterministic strided task sequence.  Runtime `num_tokens_padded` is the
   termination source, so CUDA-Graph replays remain safe when routing changes.
2. Preserve exactly-once tile coverage.  Producer and consumer roles must
   derive the same `(m_block, n_block, k_tile)` sequence or consume a shared
   mailbox; neither role may independently claim work from a racing counter.
3. Use a dedicated producer warpgroup for packed-weight TMA and cooperative
   activation gathering, and one 128-thread consumer warpgroup for RS-WGMMA.
   Keep accumulator registers out of the producer path.
4. Initialize transaction-full and consumer-empty barriers once.  A stage is
   reusable only after the consumer has completed every shared-memory read;
   termination must not leave either role waiting for an arrival that will
   never occur.
5. A transaction-full stage needs both conditions: the TMA byte transaction
   must complete and ordinary shared-memory activation/metadata writes must be
   published.  Model these as separate arrivals on the same full barrier.
6. Retain a 128-channel output tile.  Prior measurements already rejected a
   global 64-channel tile and a two-independent-task 256-thread CTA; the new
   experiment tests role specialization and cross-task overlap, not either
   rejected design.
7. Benchmark only complete TP4 CUDA Graphs, with a separate 256 MiB L2 clear
   before every timed replay and the clear outside the CUDA-event interval.
   TP8 must pass numerical and graph run-through checks before acceptance.
