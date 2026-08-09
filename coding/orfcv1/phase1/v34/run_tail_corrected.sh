#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${1:?usage: run_tail_corrected.sh RUN_ID}
ROOT_DIR=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
PY=/home/user/anaconda3/envs/featcodec2/bin/python
OLD=$ROOT_DIR/results/dinov2_vitl14/phase1/v34/v34_twosided_formal_20260809T074819Z
ROOT=$ROOT_DIR/results/dinov2_vitl14/phase1/v34/$RUN_ID
LOG=$ROOT_DIR/logs/$RUN_ID
mkdir -p "$ROOT" "$LOG"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run_one () {
  local block=$1 arm=$2 gpu=$3 init_arm=opq
  [[ $arm == identity_fixed ]] && init_arm=identity
  local out=$ROOT/$block/$arm
  local eval=(--image-batch 16 --chunk 16)
  [[ $block == blk05 ]] && eval=(--image-batch 8 --chunk 4)
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m phase1.v34.tail_corrected train \
    --block "$block" --arm "$arm" --device cuda \
    --init-codec "$OLD/$block/$init_arm/codec.pt" --output "$out" \
    --epochs 100 --batch 32 --lr 3e-4 --tau-start 0.5 --tau-end 0.005 \
    >"$LOG/${block}_${arm}_train.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m phase1.v34.tail_corrected evaluate \
    --block "$block" --arm "$arm" --device cuda --codec "$out/codec.pt" \
    --output "$out" --cost-images 500 --val-images 500 \
    --allocations 256 --token-pairs 512 "${eval[@]}" \
    >"$LOG/${block}_${arm}_eval.log" 2>&1
}

arms=(identity_fixed opq_fixed tail_u tail_v tail_vu)
pids=()
for i in 0 1 2 3 4; do run_one blk05 "${arms[$i]}" "$i" & pids+=("$!"); done
for i in 0 1 2; do run_one blk20 "${arms[$i]}" "$((i+5))" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
run_one blk20 tail_v 5 & pids+=("$!")
run_one blk20 tail_vu 6 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" - "$ROOT" >"$LOG/summary.json" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/*/result.json")):
    d = json.loads(path.read_text())
    s, i = d["separable_allocation"], d["interaction"]
    rows.append({"block": d["block"], "arm": d["arm"],
                 "token_pair": i["token_normalized"],
                 "channel_pair": i["channel_normalized"],
                 "spearman": s["spearman"], "delta_nrmse": s["delta_nrmse"],
                 "selected_rel": s["predicted_best_true_mse"] /
                                 s["uniform_true_mse"] - 1})
print(json.dumps(rows, indent=2))
PY
echo complete >"$LOG/driver.status"
