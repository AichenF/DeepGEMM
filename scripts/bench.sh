#!/bin/bash
# Custom AKO bench for the distributed TP MXFP4 MoE. Run on H20-GPU-06 in the
# worktree: `bash scripts/bench.sh iter-N`. Reports COMPILED/CORRECT/RUNTIME/
# REF_RUNTIME/SPEEDUP for the AKO loop to read.
set -e
LABEL="${1:-run}"
cd /home/xutingz/fac/DeepGEMM_tp

export DG_JIT_CACHE_DIR=/tmp/dg_jit_h20          # home NFS quota is full
export MASTER_PORT=$(( 8600 + (RANDOM % 350) ))  # unique-ish port (os._exit skips TCPStore cleanup)
M="${M:-8}"

python bench/tp_bench.py --solution solution/tp_moe_kernel.py --m "$M" 2>&1 | tee _bench_output.txt

# trajectory snapshot
TRAJ_DIR="trajectory/${LABEL}"
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/" 2>/dev/null || true
cp _bench_output.txt "$TRAJ_DIR/" 2>/dev/null || true
