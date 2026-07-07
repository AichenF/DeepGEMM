# Optimization 0 - gpt5 (load_inline)

Source: repair_0_gpt5

**PREREQUISITE**: All unit tests must pass before optimization. Verify first with run-kernel.

## Directory Structure

- `*.cu` - CUDA kernel (copied from source, must pass tests)
- `*.py` - Python driver (copied from source)
- `instructions.txt` - References loaded via read-instruction (optional)
- `benchmark_results.txt` - Latency comparison (before/after)
- `profile_summary.json` - Profiling data
- `bottleneck_analysis.txt` - Identified bottlenecks and optimization strategy

## Workflow

1. Verify correctness first:
   ```bash
   python .claude/skills/run-kernel/scripts/run_kernel.py \
       --cu kernel.cu --py driver.py
   ```

2. Profile:
   ```bash
   python .claude/skills/run-kernel/scripts/run_kernel.py \
       --cu kernel.cu --py driver.py --profile
   ```

3. Analyze bottlenecks from profile_summary.json

4. Apply optimization based on bottleneck analysis

5. Re-verify correctness (MANDATORY):
   ```bash
   python .claude/skills/run-kernel/scripts/run_kernel.py \
       --cu kernel.cu --py driver.py
   ```

6. Compare benchmark results

7. If needed, create optimize_{index+1}_{model} and iterate
