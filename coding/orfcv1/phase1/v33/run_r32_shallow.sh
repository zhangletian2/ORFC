#!/usr/bin/env bash
# blk05 / blk10 / blk15 @ R32: lambda=0 ORFC baselines plus the three v33 arms.
#
#   ./run_r32_shallow.sh init
#   CUDA_VISIBLE_DEVICES=N nohup ./run_r32_shallow.sh worker N &
#
# blk20 R32 already ran and showed the rate is where blk20's representation
# collapses outright (CLS and patch relative error both past 1.0).  The shallow
# blocks are the test of whether R32 itself is viable or only blk20 was: their
# init ladders drop a group for 1.6x the error rather than blk20's 3.1x, so the
# search has cheaper room to work with.
#
# Hyperparameters track run_sweep_ep150.sh exactly -- 50-epoch supernet, 5000
# search evals, 150-epoch delivery, batch 32, no L bank, U0 trainable in phase
# 2 -- so the R32 column reads against the existing cells.
#
# One shared queue rather than fixed per-GPU chains: blk05's tail is 19 blocks
# against blk15's 9, so static assignment would idle cards.  Seeded longest
# first, the usual LPT bound.
set -u

ROOT=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
ORFC_DIR=/data4/workspace/zlt/featcodec/coding/vq/v3.4
PY=/home/user/anaconda3/envs/featcodec2/bin/python
RES=$ROOT/results/dinov2_vitl14/phase1/v33
SWEEP=$RES/r32_shallow_20260809
LOGS=$SWEEP/logs
QUEUE=$SWEEP/queue.txt
LOCK=$SWEEP/queue.lock
PROGRESS=$SWEEP/progress.log

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

note () {
    printf '[%s][gpu%s] %s\n' "$(date +%H:%M:%S)" "${GPU:-?}" "$*" \
        | tee -a "$PROGRESS"
}

# --- queue primitives ---------------------------------------------------------
pop_job () {
    local job=""
    exec 9>"$LOCK"; flock 9
    job=$(head -n 1 "$QUEUE" 2>/dev/null || true)
    if [[ -n "$job" ]]; then
        tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
    fi
    flock -u 9; exec 9>&-
    printf '%s' "$job"
}

push_front () {
    exec 9>"$LOCK"; flock 9
    { printf '%s\n' "$1"; cat "$QUEUE"; } > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
    flock -u 9; exec 9>&-
}

# --- job bodies ---------------------------------------------------------------
# orfc:BLK:EPOCHS -- K=2 is one bit per group, the R32 uniform point.
run_orfc () {
    local blk=$1 epochs=$2
    local tag=orfc_${blk}_K2_ep${epochs}_lmbda0
    local pt=$ORFC_DIR/checkpoints/dinov2_vitl14/${blk}_K2_emb32_bt1024_ws_tau0.5_lr0.0003_ep${epochs}_n5000_s42.pt
    if [[ -f $pt ]]; then note "skip $tag (exists)"; return 0; fi
    note "start $tag"
    cd "$ORFC_DIR" || return 1
    "$PY" -u run_soft_pq.py \
        --layer "$blk" --K 2 --epochs "$epochs" \
        --embedding_dim 32 --bottleneck_dim 1024 --warm_start_opq \
        --lmbda 0.0 --tau_start 0.5 --tau_end 0.005 \
        --lr 3e-4 --max_train_images 5000 --seed 42 \
        > >(tee "$LOGS/$tag.log") 2>&1
    note "done  $tag"
}

# p2:BLK:ARM:ALLOCATION -- 150-epoch specialised delivery.
run_p2 () {
    local blk=$1 arm=$2 alloc=$3
    local tag=v33_ep150_${blk}_R32_${arm}
    if [[ -f $RES/$tag/R32/checkpoint.pt ]]; then
        note "skip $tag (exists)"; return 0
    fi
    note "start $tag"
    cd "$ROOT" || return 1
    "$PY" -u -m phase1.v33.train \
        --phase 2 --block "$blk" --anchor R32 --device cuda --run-id "$tag" \
        --source "$RES/v33_r32_$blk/R32" --allocation "$alloc" \
        --epochs 150 --batch 32 --ckpt-every 1000 --no-freeze-u0 --no-L \
        > >(tee "$LOGS/$tag.log") 2>&1
    note "done  $tag"
}

# pipe:BLK -- phase 1 -> search -> searched arm; uniform goes back on the queue.
run_pipe () {
    local blk=$1
    local run=v33_r32_$blk
    local out=$RES/$run/R32
    cd "$ROOT" || return 1

    if [[ ! -f $out/checkpoint.pt ]]; then
        note "start $run phase1"
        "$PY" -u -m phase1.v33.train \
            --phase 1 --block "$blk" --anchor R32 --device cuda \
            --run-id "$run" --epochs 50 --batch 32 --ckpt-every 500 --no-L \
            > >(tee "$LOGS/${run}_phase1.log") 2>&1
    fi
    [[ -f $out/checkpoint.pt ]] || { note "FAIL $run phase1"; return 1; }

    if [[ ! -f $out/uniform_allocation.npy ]]; then
        "$PY" - "$blk" "$out/uniform_allocation.npy" <<'PY'
import sys
import numpy as np
from phase1.v21.config import activate
from phase1.v12 import config as C
from phase1 import engine
activate(sys.argv[1])
np.save(sys.argv[2], np.asarray(
    engine.uniform_allocation(C.ANCHOR_BY_NAME["R32"], C.GROUPS), dtype=np.int64))
PY
    fi

    # The uniform control needs only phase 1; hand it to whichever card frees up.
    push_front "p2:$blk:uniform:$out/uniform_allocation.npy"

    if [[ ! -f $out/allocation.npy ]]; then
        note "start $run search"
        "$PY" -u -m phase1.v33.search \
            --block "$blk" --anchor R32 --device cuda \
            --checkpoint "$out/checkpoint.pt" \
            --body-images 128 --recheck-images 500 --eval-budget 5000 \
            --propose-top 50 --out "$out/search.json" \
            > >(tee "$LOGS/${run}_search.log") 2>&1
    fi
    [[ -f $out/allocation.npy ]] || { note "FAIL $run search"; return 1; }

    run_p2 "$blk" searched "$out/allocation.npy"
}

# --- entry points -------------------------------------------------------------
case "${1:?usage: run_r32_shallow.sh init | worker GPU}" in
init)
    mkdir -p "$LOGS"
    cat > "$QUEUE" <<'EOF'
pipe:blk05
orfc:blk05:300
pipe:blk10
pipe:blk15
orfc:blk10:300
orfc:blk15:300
orfc:blk05:100
orfc:blk10:100
orfc:blk15:100
EOF
    printf 'queued %d jobs in %s\n' "$(wc -l < "$QUEUE")" "$QUEUE"
    ;;
worker)
    GPU=${2:?worker needs a GPU index}
    note "worker up"
    while :; do
        job=$(pop_job)
        [[ -z "$job" ]] && break
        IFS=: read -r kind a b c <<< "$job"
        case "$kind" in
        orfc) run_orfc "$a" "$b" ;;
        p2)   run_p2 "$a" "$b" "$c" ;;
        pipe) run_pipe "$a" ;;
        *)    note "unknown job $job" ;;
        esac
    done
    note "worker done (queue empty)"
    ;;
*)
    echo "usage: run_r32_shallow.sh init | worker GPU" >&2
    exit 1
    ;;
esac
