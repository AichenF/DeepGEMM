#!/bin/bash
# ako4x parallel config sweep: run candidate (nA,nB,threads) tilings in parallel
# across the free GPUs (0-4) inside the bench container. No recompile per config.
WT=/lustre/raplab/client/xutingz/fac/DeepGEMM_tp
C=mlb_refactor_bench_20260809
M=${M:-1}
shift 2>/dev/null || true
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=("8 16 256" "16 24 256" "16 48 256" "32 48 256" "16 48 128" "16 96 256" "32 96 256" "16 192 256" "64 96 256" "16 384 256")
rm -f /tmp/ako4x_res_*.txt
i=0
for cfg in "${CONFIGS[@]}"; do
  gpu=$((i % 5))
  set -- $cfg; nA=$1; nB=$2; th=$3
  ( r=$(docker exec $C bash -lc "cd $WT && TORCH_EXTENSIONS_DIR=/tmp/torch_ext_tp TP_REPO=$WT CUDA_VISIBLE_DEVICES=$gpu TP_NA=$nA TP_NB=$nB TP_THREADS=$th python bench/tp_compute_bench.py --solution solution/tp_moe_kernel.py --m $M 2>/dev/null" | grep -oE 'RUNTIME=[0-9.]+|CORRECT=[A-Za-z]+')
    echo "nA=$nA nB=$nB th=$th gpu=$gpu :: $(echo $r | tr '\n' ' ')" > /tmp/ako4x_res_$i.txt ) &
  i=$((i+1))
  [ $((i % 5)) -eq 0 ] && wait
done
wait
echo "=== M=$M sweep results (sorted by RUNTIME) ==="
cat /tmp/ako4x_res_*.txt | sort -t= -k4 -g
