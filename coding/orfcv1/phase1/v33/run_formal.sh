#!/usr/bin/env bash
# V33 formal pipeline: blk20-R64, OPQ init → phase1 → search → phase2 → verify.
# Budget reference: ~50 epoch fair supernet + ~75 epoch specialised delivery.
set -euo pipefail

ROOT="/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1"
PY="/home/user/anaconda3/envs/featcodec2/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"

RUN_ID="${RUN_ID:-v33_formal_20260807T050000Z}"
ANCHOR="${ANCHOR:-R64}"
DEVICE="${DEVICE:-cuda}"
P1_EPOCHS="${P1_EPOCHS:-50}"
P2_EPOCHS="${P2_EPOCHS:-75}"
CKPT_EVERY="${CKPT_EVERY:-250}"
# Optional: pin the phase-1 -> search hand-off to a specific step.
SWITCH_STEP="${SWITCH_STEP:-}"

OUT="results/dinov2_vitl14/phase1/v33/${RUN_ID}/${ANCHOR}"
P1_OUT="$OUT"
P2_RUN_ID="${RUN_ID}_p2"
P2_OUT="results/dinov2_vitl14/phase1/v33/${P2_RUN_ID}/${ANCHOR}"
# Keep logs outside train output dirs (train refuses non-empty run roots).
LOG_DIR="results/dinov2_vitl14/phase1/v33/${RUN_ID}_pipeline_logs"
mkdir -p "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_DIR/pipeline.log"; }

log "=== V33 formal start RUN_ID=$RUN_ID ANCHOR=$ANCHOR GPU=$CUDA_VISIBLE_DEVICES ==="

# --- Phase 1 -----------------------------------------------------------------
if [[ ! -f "$P1_OUT/checkpoint.pt" ]]; then
  log "Phase 1: epochs=$P1_EPOCHS ckpt_every=$CKPT_EVERY"
  "$PY" -u -m phase1.v33.train \
    --phase 1 --anchor "$ANCHOR" --run-id "$RUN_ID" --device "$DEVICE" \
    --epochs "$P1_EPOCHS" --ckpt-every "$CKPT_EVERY" \
    2>&1 | tee "$LOG_DIR/phase1.log"
else
  log "Phase 1 checkpoint exists; skipping train"
fi

if [[ ! -f "$P1_OUT/checkpoint.pt" ]]; then
  log "ERROR: phase 1 produced no final checkpoint under $P1_OUT"
  exit 1
fi

# --- Noise floor on the final checkpoint -------------------------------------
NOISE_JSON="$OUT/noise.json"
if [[ ! -f "$NOISE_JSON" ]]; then
  log "Noise floor on $P1_OUT/checkpoint.pt"
  "$PY" -u -m phase1.v33.noise_floor \
    --checkpoint "$P1_OUT/checkpoint.pt" --anchor "$ANCHOR" --device "$DEVICE" \
    --out "$NOISE_JSON" \
    2>&1 | tee "$LOG_DIR/noise_floor.log"
else
  log "Noise floor exists; skipping"
fi

# --- Pick switch checkpoint --------------------------------------------------
# The probe is retired.  On the 7800-step R64 run it burned ~116 min to report
# Kendall-tau == 1.0 and Hamming == 0 at every pair, i.e. it never resolved a
# transition and still nominated step 1000.  Search on the end of phase 1
# instead: it is the most-trained supernet and costs nothing to identify.
SWITCH_META="$OUT/switch_choice.json"
SWITCH_CKPT="$P1_OUT/checkpoint.pt"
if [[ -n "${SWITCH_STEP:-}" ]]; then
  SWITCH_CKPT="$(printf '%s/checkpoint_step%06d.pt' "$P1_OUT" "$SWITCH_STEP")"
  [[ -f "$SWITCH_CKPT" ]] || { log "ERROR: no checkpoint at step $SWITCH_STEP"; exit 1; }
fi
"$PY" - "$SWITCH_CKPT" "$SWITCH_META" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[2]).write_text(json.dumps({
    "checkpoint": str(Path(sys.argv[1]).resolve()),
    "reason": "end_of_phase1" if "step" not in Path(sys.argv[1]).stem
              else "SWITCH_STEP override",
}, indent=2))
PY
log "Switch checkpoint: $SWITCH_CKPT (see $SWITCH_META)"

# --- Search ------------------------------------------------------------------
SEARCH_JSON="$OUT/search.json"
ALLOC_NPY="$OUT/allocation.npy"
if [[ ! -f "$ALLOC_NPY" ]]; then
  log "Search on switch checkpoint"
  "$PY" -u -m phase1.v33.search \
    --checkpoint "$SWITCH_CKPT" --anchor "$ANCHOR" --device "$DEVICE" \
    --body-images 128 --recheck-images 500 --eval-budget 5000 \
    --propose-top 50 --out "$SEARCH_JSON" \
    2>&1 | tee "$LOG_DIR/search.log"
  # search writes allocation.npy next to --out
  if [[ -f "${SEARCH_JSON%.json}.npy" && ! -f "$ALLOC_NPY" ]]; then
    cp "${SEARCH_JSON%.json}.npy" "$ALLOC_NPY"
  fi
  if [[ -f "$(dirname "$SEARCH_JSON")/allocation.npy" && "$OUT/allocation.npy" != "$(dirname "$SEARCH_JSON")/allocation.npy" ]]; then
    cp "$(dirname "$SEARCH_JSON")/allocation.npy" "$ALLOC_NPY" || true
  fi
  # Prefer explicit path written by search module
  if [[ ! -f "$ALLOC_NPY" ]]; then
    found="$(ls -1 "$OUT"/*allocation*.npy 2>/dev/null | head -1 || true)"
    if [[ -n "${found:-}" ]]; then
      cp "$found" "$ALLOC_NPY"
    fi
  fi
  if [[ ! -f "$ALLOC_NPY" ]]; then
    log "ERROR: allocation.npy missing after search"
    ls -la "$OUT" | tee -a "$LOG_DIR/pipeline.log"
    exit 1
  fi
else
  log "Allocation exists; skipping search"
fi

# --- Phase 2 -----------------------------------------------------------------
if [[ ! -f "$P2_OUT/checkpoint.pt" ]]; then
  log "Phase 2: epochs=$P2_EPOCHS source=$SWITCH_CKPT alloc=$ALLOC_NPY"
  "$PY" -u -m phase1.v33.train \
    --phase 2 --anchor "$ANCHOR" --run-id "$P2_RUN_ID" --device "$DEVICE" \
    --source "$SWITCH_CKPT" --allocation "$ALLOC_NPY" \
    --epochs "$P2_EPOCHS" --ckpt-every "$CKPT_EVERY" \
    2>&1 | tee "$LOG_DIR/phase2.log"
else
  log "Phase 2 checkpoint exists; skipping train"
fi

# --- Verify ------------------------------------------------------------------
VERIFY_JSON="$OUT/verify.json"
log "Verify delivery"
"$PY" -u -m phase1.v33.verify \
  --checkpoint "$P2_OUT/checkpoint.pt" \
  --allocation "$ALLOC_NPY" \
  --anchor "$ANCHOR" --device "$DEVICE" \
  --out "$VERIFY_JSON" \
  2>&1 | tee "$LOG_DIR/verify.log"

log "=== V33 formal done ==="
log "verify: $VERIFY_JSON"
log "p2: $P2_OUT/checkpoint.pt"
cat "$VERIFY_JSON" | "$PY" -c 'import json,sys; r=json.load(sys.stdin); print(json.dumps({k:r.get(k) for k in ("passed","one_opt","orfc","rate","seconds") if k in r or True}, indent=2, default=str))' \
  2>/dev/null | tee -a "$LOG_DIR/pipeline.log" || true
