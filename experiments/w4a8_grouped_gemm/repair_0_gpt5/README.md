# Repair 0 - gpt5 (load_inline)

Source: generation_0_gpt5

## Directory Structure

- `*.cu` - Current failing CUDA kernel (copied from source)
- `*.py` - Python driver (copied from source)
- `error_info.txt` - Structured error information (REQUIRED)

## Workflow

1. Save error information: `python ../../.claude/skills/repair-kernel/scripts/save_error_info.py error_info.txt ...`
2. Optionally load references via `read-instruction` skill for the specific error type
3. Agent applies fix directly to kernel.cu
4. Verify: `python .claude/skills/run-kernel/scripts/run_kernel.py --cu kernel.cu --py *.py`
5. If still failing, create repair_{index+1}_{model} and iterate
