# DSR1 preliminary full-model accuracy evidence

This is a completed 1316-question warmup round, not a completed formal timed E2E protocol.

The independently recomputed result is:

- correct: `1266/1316`
- exact accuracy: `0.9620060790273556`
- invalid outputs: `0`
- known corruption signatures: `0`
- preregistered gate: at least `1260/1316`

All eight H200 GPU UUIDs were occupied by the owned service container. The server recorded exactly `8 × 58 = 464` SM90 NVFP4 layer builds, covering layers 3 through 60 on every rank with `block_n=256` and `grouped_nibbles=True`; EPLB was disabled.

After the complete warmup and its round validator passed, the runner could not observe a DeepGEMM JIT file in the selected host cache directory. It exited at that evidence gate before the timed round. Therefore:

- the preliminary accuracy regression passes;
- the archived run as a whole remains `EXECUTION_FAIL / NOT_COMPLETED`;
- no timed/formal result exists;
- the warmup latency `363.423 s` must not be used as a formal performance result.

The fixed SGLang image used CUDA 13.0.1/torch cu130 and raw DeepGEMM `75186dd`. This does not prove that the CUDA 13.2 patched binary passed a full-model E2E run; the CUDA 13.2 patch is syntax-only, but the build identities remain distinct.

## Independent replay

The frozen GSM8K file is not duplicated here. Fetch this exact file and verify its hash:

```bash
curl -L https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl -o gsm8k_test.jsonl
sha256sum gsm8k_test.jsonl
# expected: 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14
```

Then decompress the archived raw output and run the frozen validator:

```bash
gzip -dk gsm8k_raw_warmup.jsonl.gz
python3 validate_dsr1_e2e.snapshot.py --validate-round warmup --run-dir .
```

The historical 8-shot harness uses the first eight test questions as demonstrations and still scores those questions. They were 8/8 correct. Removing them after the fact gives `1258/1308 = 0.9617737`; this is a regression proxy, not a leak-free official GSM8K score.
