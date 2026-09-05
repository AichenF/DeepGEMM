# Iteration 653a: M128 SMID trace session result unavailable

## Intent

Run the existing default-off `V4_SINGLE_LAUNCH_TRACE_SMID=1` diagnostic on
physical H20 GPU 1 for the selected TP4 M128 production kernel, after the
fixture's separate excluded 256 MiB L2 clear.  The intended checks were full
bitwise `down` equality against the same-source multi-kernel local reference,
78-SM x 9-CTA placement, and the per-SM distribution of the 474 W13 residual
tasks.  No performance timing was intended.

## Client result

The remote command entered JIT with unified exec session `89648`, but the
client-side polling calls returned no stdout and subsequently reported
`Unknown process id 89648`.  No terminal exit code, profile JSON, correctness
record, or tail-map record was delivered to the client in this iteration.

The command was configured to tee remote stdout to
`bench/results/iter653_production_m128_smid_trace.log`; that file has not yet
been inspected, because this checkpoint records the failed session boundary
before issuing another diagnostic command.

## Decision

Treat the trace as unqualified: no correctness, placement, or scheduling
claim is made.  Inspect the remote log read-only next.  If it contains a
complete accepted record, derive the tail map from it as a separate
iteration; otherwise rerun the exact diagnostic with a stable outer timeout
and log-first retrieval.
