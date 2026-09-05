# Iteration 667: bulk-reduce phase-boundary isolation

## Question

The multi-kernel audit left one potentially useful epilogue adaptation:
replace W2's materialized BF16 route tensor plus terminal fixed-k6 reread with
producer-owned FP32 shared staging and Hopper
`cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`.  Earlier
monolithic integrations hung, even though a simple standalone instruction
probe passed.  This iteration isolates which multi-kernel property is
actually required for forward progress.

## SASS and address-pattern probes

The compiled monolithic W2 epilogue lowers the PTX operation to
`UBLKRED.G.S.ADD.F32.RN`, preceded by `R2UR` because its destination token is
loaded dynamically.  The original microprobe used only uniform address
arithmetic.  The probe now adds all of the missing address/lifetime cases:

- destination rows loaded from a global index table, forcing the same
  regular-to-uniform address transfer;
- 624 CTAs issuing eight reductions in opposite/permuted destination order;
- the same 624-CTA persistent grid first clearing the destination with
  generic stores, publishing every writer through
  `fence.proxy.async.global`, rendezvousing through a software whole-grid
  barrier, then issuing the reductions.

All cases completed on H20 and matched exactly (`max_error=0`).  In
particular, 624 CTAs x eight crossing destinations and the in-kernel-zero
variant both passed.  This rejects dynamic `R2UR` address formation,
destination lock order/contention, and same-kernel destination initialization
as causes of the integration hang.

Raw artifacts:

- `bench/results/iter667_cp_reduce_dynamic_destination_probe_20260906.log`
- `bench/results/iter667b_cp_reduce_in_kernel_zero_probe_20260906.log`

## Exact W2 isolation

A temporary diagnostic entry instantiated the same
`route_gemm_task<512,4096,1,false>` TMA, MXFP4 decode, WGMMA, and bulk-reduce
epilogue outside the complete MegaMoE entry.

- One fresh CTA per W2 task completed at random-route M8.  Against an
  independent BF16-route-output fixed-k6 reference, cosine was
  `0.999999999999997`, relative L2 `8.36e-8`, and max absolute difference
  `0.01171875`.
- A 624-CTA grid-stride version, in which each persistent CTA executed
  multiple W2 tasks and multiple commit/wait groups exactly like the
  monolithic W2 phase, also completed.  Its different inter-CTA FP32 add
  order produced cosine `0.999999749989`, relative L2 `7.0719e-4`, and finite
  output; this is a forward-progress isolation result, not a selected
  numerical/performance result.

Both target calls had a separate excluded 256 MiB L2 clear.  These results
reject preceding W2 TMA/WGMMA, repeated bulk groups, and persistent W2 task
reuse as causes.

Raw artifacts:

- `bench/results/iter667c_standalone_w2_bulk_route1_20260906.log`
- `bench/results/iter667d_persistent_w2_bulk_route1_20260906.log`

## Complete-entry negative controls

The complete one-kernel M8 path was then restricted so only CTA0 could issue
one valid 512-byte reduction after the ordinary route, W13, requant, and W2
phases.  It still failed to return.  Explicitly invalidating every completed
W2 TMA stage mbarrier before entering the bulk async proxy also failed to
restore progress.  Both processes were terminated after confirming the live
GPU-side stall; neither produced a correctness or timing result.

Raw artifacts:

- `bench/results/iter667e_monolithic_bulk_cta0_m8_20260906.log`
- `bench/results/iter667f_monolithic_bulk_cta0_mbar_inval_m8_20260906.log`

## Verdict

The only condition correlated with failure is crossing the earlier W13 and
whole-grid phase state in the same resident CTA before the W2 bulk epilogue.
The multi-kernel version obtains a real kernel-exit/relaunch async-state
boundary; it is not equivalent to a source-level barrier, proxy fence,
mbarrier invalidation, fresh bulk group, or fresh W2 loop.  Since even one
issuer hangs, throttling bulk concurrency cannot make this adaptation safe.

Reject producer-side async bulk combine for the one-kernel path.  Keep its
flag default-off, retain the expanded standalone probe as regression evidence,
and restore the production CUDA/Python source byte-for-byte to the selected
head.  No latency improvement is claimed in this iteration.
