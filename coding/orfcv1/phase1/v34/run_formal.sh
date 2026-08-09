#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${1:?usage: run_formal.sh RUN_ID}
CONDA=/home/user/anaconda3/bin/conda
ROOT=results/dinov2_vitl14/phase1/v34/$RUN_ID
LOG=logs/$RUN_ID
mkdir -p "$ROOT" "$LOG"

run_probe() {
  local block=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu "$CONDA" run -n featcodec2 --no-capture-output \
    python -m phase1.v34.run probe --block "$block" --device cuda \
    --output "$ROOT/$block/probe.npz" >"$LOG/${block}_probe.log" 2>&1
}
run_arm() {
  local block=$1 arm=$2 gpu=$3
  local eval_args=(--image-batch 16 --candidate-chunk 16)
  if [[ $block == blk05 ]]; then
    eval_args=(--image-batch 8 --candidate-chunk 4)
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$CONDA" run -n featcodec2 --no-capture-output \
    python -m phase1.v34.run arm --block "$block" --arm "$arm" --device cuda \
    --probe "$ROOT/$block/probe.npz" --output "$ROOT/$block/$arm" \
    "${eval_args[@]}" \
    >"$LOG/${block}_${arm}.log" 2>&1
}

run_probe blk05 0 & p0=$!
run_probe blk20 1 & p1=$!
wait "$p0" "$p1"

arms=(identity opq task_u task_v task_vu)
pids=()
for index in 0 1 2 3 4; do run_arm blk05 "${arms[$index]}" "$index" & pids+=("$!"); done
for index in 0 1 2; do run_arm blk20 "${arms[$index]}" "$((index+5))" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done

pids=()
run_arm blk20 task_v 5 & pids+=("$!")
run_arm blk20 task_vu 6 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

python - <<'PY' "$ROOT" >"$LOG/summary.json"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/*/result.json")):
    data = json.loads(path.read_text())
    rows.append({
        "block": data["block"], "arm": data["arm"],
        "token_abs_rel": data["coupling"]["token_residual"]["mean_abs_over_full"],
        "channel_abs_rel": data["coupling"]["channel_residual"]["mean_abs_over_full"],
        "spearman": data["separable_allocation"]["spearman"],
        "nrmse": data["separable_allocation"]["nrmse"],
        "delta_nrmse": data["separable_allocation"]["delta_nrmse"],
        "selected_gain": (data["separable_allocation"]["uniform_true_mse"] -
                          data["separable_allocation"]["predicted_best_true_mse"]),
    })
print(json.dumps(rows, indent=2))
PY
echo complete >"$LOG/driver.status"
