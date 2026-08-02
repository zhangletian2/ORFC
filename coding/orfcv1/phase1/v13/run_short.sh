#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:?usage: run_short.sh RUN_ID}"
STEPS="${STEPS:-500}"
ROOT="results/dinov2_vitl14/phase1/v13/${RUN_ID}"
LOGS="${ROOT}/logs"
mkdir -p "${LOGS}"

common=(--device cuda --epochs 100 --steps "${STEPS}" --batch 32
  --num-workers 4 --optimizer-mode orfc_adam --lr-u 3e-4 --lr-theta 3e-4
  --policy-gradient reinforce --policy-samples 4 --policy-lr 1e-2
  --temperature 1.0 --entropy-weight 1e-2
  --codeword-temperature 0.5 --codeword-temperature-end 0.005)

launch() {
  local gpu="$1" anchor="$2" arm="$3"; shift 3
  CUDA_VISIBLE_DEVICES="${gpu}" python -m phase1.v12.train \
    --anchor "${anchor}" --run-id "${RUN_ID}_${arm}" \
    "${common[@]}" "$@" >"${LOGS}/${arm}_${anchor}.log" 2>&1 &
  pids+=("$!")
}

pids=()
launch 2 R64 baseline
launch 3 R96 baseline
launch 4 R64 coverage --coverage-samples 1 --coverage-weight 0.2
launch 6 R96 coverage --coverage-samples 1 --coverage-weight 0.2
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if (( status )); then
  echo "one or more v13 short arms failed; inspect ${LOGS}" >&2
  exit 1
fi

python -m phase1.v13.audit_short \
  --baseline-run "${RUN_ID}_baseline" \
  --coverage-run "${RUN_ID}_coverage" \
  --output "${ROOT}/audit.json"
