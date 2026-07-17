# Flash/Pro formal-run stability preflight

These files are health gates, not performance samples.

Before restarting the final Flash/Pro 30-process protocol, the node passed:

1. all 56 directed mapped-P2P read/write pairs (the formal record is `../final_flash_pro_mapped_p2p.log`);
2. Aichen `ba7ee094` baseline, Pro H=7168/I=3072/E=384/top-k=6, M=8192, `--num-tests 500`, exit 0;
3. sealed final `75186dd + CUDA 13.2 16-edit patch`, the same Pro M=8192 workload and `--num-tests 500`, exit 0.

Each `500` gate is one fresh benchmark process containing 500 kernel timing iterations. It is not 500 independent processes and is excluded from every median30 table. The two subdirectories retain the exact image, source hashes, protocol, start/end time, GPU before/after snapshots, exit code and raw stress log.

The formal protocol then restarted from its excluded warmups and round 1. No sample from the earlier invalid 19-round attempt was reused.
