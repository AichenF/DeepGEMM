# Generation 0 - gpt5 (load_inline)

## Directory Structure

- `*.py` - Python driver (contains ReferenceModel, CUSTOM_MODEL_CPP_BINDINGS, main)
- `instructions.txt` - Combined instruction content
- `prompt.txt` - Complete generation prompt
- `llm_response.md` - LLM response
- `kernel.cu` - Extracted CUDA kernel

## Workflow

1. Place or copy the Python driver `.py` in this directory
2. Save instructions: `python ../../scripts/save_instructions.py instructions.txt ...`
3. Generate prompt: `python ../../scripts/generate_prompt.py --mode load_inline driver.py instructions.txt > prompt.txt`
4. Send prompt to LLM, save response as `llm_response.md`
5. Extract kernel: `python ../../scripts/extract_code.py kernel.cu llm_response.md --lang cuda`
6. Verify: Use `run-kernel` skill with `--cu kernel.cu --py driver.py`
