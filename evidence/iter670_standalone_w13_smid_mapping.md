# Iteration 670 — standalone W13 task-to-SM mapping is replay-dynamic

## Question

Could the multi-kernel W13 launch's fresh-CTA task placement be measured once
and transferred as a static permutation to the persistent one-kernel W13
phase?

## Method

- Added temporary, default-off instrumentation to the exact standalone
  `route_gemm<4096,1024,2,true>` specialization. Lane 0 recorded `%smid` once
  at CTA entry.
- Ran three M128 standalone W13 launches on one H20. Each measured launch used
  a separate excluded 256 MiB L2 clear.
- Analyzed only the first 3,984 valid W13 tasks (1,992 padded rows); the launch
  has 5,136 grid slots because its expert buffer is capacity-sized.
- Compared the traces with the selected single-kernel M128 placement recorded
  in Iteration 653 (702 resident CTAs, exactly nine CTAs per each of 78 SMs).

## Results

- Standalone elapsed times were 274.592, 189.248 and 189.888 us. The first is
  an instrumentation/initial-run outlier; timing is not used for selection.
- All three task-to-SM trace hashes differ.
- Pairwise exact task-owner agreement over 3,984 valid tasks is only 2.836%,
  1.054% and 1.682%.
- Per 702-task full wave, pairwise exact agreement ranges from 0% to 7.977%.
- Despite different identities, every full wave remains balanced: every SM
  receives 8--10 tasks. The residual 474-task wave gives every SM 5--7 tasks.
- The same-ordinal task retains its owner across adjacent standalone waves
  only 1.14%--8.12% of the time. By comparison, the unrotated persistent
  mapping retains 100%; both selected nonzero wave rotations reduce measured
  persistent adjacent-wave owner overlap to 0%.
- Fitting one circular permutation to the first standalone wave can match
  62.1%--75.5% for that replay, but its fit collapses to about 12% on wave 2
  and about 3% thereafter. Best shifts also differ by replay and wave.

## Verdict

Reject a recorded static lookup/permutation. The standalone mapping is
completion-driven and not replay-stable; copying one trace would overfit an
instrumented launch. The transferable property is balanced per-wave work with
owner decorrelation, already approximated by the selected W13 shift-13 and W2
shift-82 rotations. This does not remove the measured fixed-entry resource and
issue-efficiency gap (`REG56` and 0.63 issued warps/scheduler/cycle versus
standalone W13 `REG47` and 0.74).

Temporary tracing code was removed and production source restored to SHA-256
`7ac22134c953d17c8dea9310011818ca483b5b8a96b06324381abd2c8c9c3f45`.

## Evidence

- `bench/results/iter670_standalone_w13_smid_trace_20260906.log`
- `bench/results/iter653_production_m128_smid_trace.log`
