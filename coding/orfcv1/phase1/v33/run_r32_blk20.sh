#!/usr/bin/env bash
# blk20 @ R32: the lambda=0 ORFC baselines plus the three v33 arms.
#
#   nohup ./run_r32_blk20.sh > /dev/null 2>&1 &
#
# R32 is one bit per group, the floor of the R64 ladder, so its modes are
# (0,1,2) and the down-mode drops a group to a single centroid.  Everything
# else -- 50-epoch supernet, 5000-eval search, 150-epoch delivery, batch 32,
# no L bank, U0 left trainable in phase 2 -- matches run_sweep_ep150.sh so the
# R32 column can be read against the existing eight cells.
#
# The two ORFC runs and the supernet start together; the uniform control is
# handed to a fourth card the moment phase 1 lands, since it only needs the
# phase-1 checkpoint and not the search result.
set -u

ROOT=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
ORFC_DIR=/data4/workspace/zlt/featcodec/coding/vq/v3.4
PY=/home/user/anaconda3/envs/featcodec2/bin/python
RES=$ROOT/results/dinov2_vitl14/phase1/v33
OUT=$RES/r32_blk20_20260808
LOGS=$OUT/logs
PROGRESS=$OUT/progress.log

RUN=v33_r32_blk20            # supernet + search live here
P1=$RES/$RUN/R32

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGS"
note () { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$PROGRESS"; }

# --- lambda=0 ORFC baselines, K=2 is 1 bit x 32 groups -----------------------
run_orfc () {
    local gpu=$1 epochs=$2
    local tag=orfc_blk20_K2_ep${epochs}_lmbda0
    local pt=$ORFC_DIR/checkpoints/dinov2_vitl14/blk20_K2_emb32_bt1024_ws_tau0.5_lr0.0003_ep${epochs}_n5000_s42.pt
    if [[ -f $pt ]]; then note "skip $tag (exists)"; return 0; fi
    note "start $tag on gpu$gpu"
    ( cd "$ORFC_DIR" && CUDA_VISIBLE_DEVICES=$gpu "$PY" -u run_soft_pq.py \
        --layer blk20 --K 2 --epochs "$epochs" \
        --embedding_dim 32 --bottleneck_dim 1024 --warm_start_opq \
        --lmbda 0.0 --tau_start 0.5 --tau_end 0.005 \
        --lr 3e-4 --max_train_images 5000 --seed 42 ) \
        > >(tee "$LOGS/$tag.log") 2>&1
    note "done  $tag"
}

# --- 150-epoch specialised delivery ------------------------------------------
run_p2 () {
    local gpu=$1 arm=$2 alloc=$3
    local tag=v33_ep150_blk20_R32_${arm}
    if [[ -f $RES/$tag/R32/checkpoint.pt ]]; then
        note "skip $tag (exists)"; return 0
    fi
    note "start $tag on gpu$gpu"
    ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=$gpu "$PY" -u -m phase1.v33.train \
        --phase 2 --block blk20 --anchor R32 --device cuda --run-id "$tag" \
        --source "$P1" --allocation "$alloc" \
        --epochs 150 --batch 32 --ckpt-every 1000 --no-freeze-u0 --no-L ) \
        > >(tee "$LOGS/$tag.log") 2>&1
    note "done  $tag"
}

note "=== blk20 R32 sweep up ==="

run_orfc 3 300 &
orfc300=$!
run_orfc 4 100 &
orfc100=$!

# --- phase 1: strict-fair supernet -------------------------------------------
if [[ ! -f $P1/checkpoint.pt ]]; then
    note "start $RUN phase1 on gpu2"
    ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=2 "$PY" -u -m phase1.v33.train \
        --phase 1 --block blk20 --anchor R32 --device cuda \
        --run-id "$RUN" --epochs 50 --batch 32 --ckpt-every 500 --no-L ) \
        > >(tee "$LOGS/${RUN}_phase1.log") 2>&1
fi
[[ -f $P1/checkpoint.pt ]] || { note "FAIL phase1"; exit 1; }
note "phase1 ready"

# The uniform control needs nothing from the search, so start it now.
if [[ ! -f $P1/uniform_allocation.npy ]]; then
    ( cd "$ROOT" && "$PY" - "$P1/uniform_allocation.npy" <<'PY'
import sys
import numpy as np
from phase1.v21.config import activate
from phase1.v12 import config as C
from phase1 import engine
activate("blk20")
np.save(sys.argv[1], np.asarray(
    engine.uniform_allocation(C.ANCHOR_BY_NAME["R32"], C.GROUPS), dtype=np.int64))
PY
    )
fi
run_p2 6 uniform "$P1/uniform_allocation.npy" &
uniform=$!

# --- search -------------------------------------------------------------------
if [[ ! -f $P1/allocation.npy ]]; then
    note "start $RUN search on gpu2"
    ( cd "$ROOT" && CUDA_VISIBLE_DEVICES=2 "$PY" -u -m phase1.v33.search \
        --block blk20 --anchor R32 --device cuda \
        --checkpoint "$P1/checkpoint.pt" \
        --body-images 128 --recheck-images 500 --eval-budget 5000 \
        --propose-top 50 --out "$P1/search.json" ) \
        > >(tee "$LOGS/${RUN}_search.log") 2>&1
fi
[[ -f $P1/allocation.npy ]] || { note "FAIL search"; exit 1; }
note "search ready"

run_p2 2 searched "$P1/allocation.npy"

wait $uniform $orfc300 $orfc100
note "=== blk20 R32 sweep complete ==="
