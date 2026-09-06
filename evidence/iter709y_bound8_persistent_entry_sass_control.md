# Iteration 709y — 64-register fused-entry W2 scheduling control

## Scope

This is a compile-only control for the selected TP4 single-launch source.  It
disables `V4_SINGLE_LAUNCH_COMPACT_W13_BUNDLE`, restoring the prior 64-register
eight-CTA launch-bound specialization, while leaving every Iteration 709l–709w
diagnostic flag at its default value of zero.  No MoE business kernel was
launched and no latency result is claimed.

The first print-only diagnostic used an unquoted label and raised `NameError`
after the extension had loaded successfully.  A cached rerun printed the exact
extension identity and path; this scripting error did not execute CUDA work or
affect the generated cubin.

## Exact build

- Environment: `CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST=9.0a`
- Overrides: `V4_SINGLE_LAUNCH_TP4=1`
  `V4_SINGLE_LAUNCH_COMPACT_W13_BUNDLE=0`
- Extension: `v4tp_8e3ee4aad14428dc50e9_v178mspec`
- Target entry: TP4 `tp4_megamoe_single_launch_kernel<2, 128>`

## Resource and SASS gate

- Resource usage: `REG64 STACK32 SHARED4096 LOCAL0`.
- Exact target SASS SHA256:
  `145c931f426a3cb8bbcbcdc8fb30cb167cd7566752040ace112a1584616dc6aa`.
- Static instruction counts: 64 QGMMAs, 64
  `WARPGROUP.DEPBAR.LE`, and 24 total `WARPSYNC` instructions.
- Inspection of the W2 regions confirms that ptxas still places a dependency
  wait after every QGMMA; the ratio is 1:1, not standalone W2's 2:1.

## Decision

**Reject as a scheduling remedy before correctness or timing.**  Raising the
fused entry from 56 to 64 registers and dropping from nine to eight resident
CTAs does not recover standalone W2's paired-QGMMA schedule.  The serialization
is therefore not explained by the 56-register capacity cliff alone; the large
persistent entry's phase-spanning lifetime/control-flow contract is the stronger
constraint.  Close further same-entry operand-order/fence/register-limit probes
unless a phase-local code-generation boundary changes that contract.

Raw artifacts:

- `bench/results/iter709y_bound8_jit.log`
- `bench/results/iter709y_bound8_resources.log`
- `bench/results/iter709y_bound8_m128_split2.sass`
