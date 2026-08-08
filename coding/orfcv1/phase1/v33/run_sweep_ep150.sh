#!/usr/bin/env bash
# Sweep: fill the missing ORFC lambda=0 baselines, extend v33 to blk10/blk15,
# and re-run every v33 phase-2 arm at 150 epochs so each search gain is read
# against a budget-matched uniform control instead of a 75-epoch one.
#
#   ./run_sweep_ep150.sh init            # build the queue
#   CUDA_VISIBLE_DEVICES=N ./run_sweep_ep150.sh worker N
#
# Workers pull from one shared queue rather than following a fixed per-GPU
# chain: the blk10/blk15 durations are extrapolated from blk05/blk20 by tail
# depth and are good to maybe +-20%, so static assignment would idle cards.
# The queue is seeded longest-job-first, which is the usual LPT bound.
#
# blk05 R64 ep300 lr3e-4 is not retrained; it already exists under the older
# naming that spells out lmbda0.0.
set -u

ROOT=/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
ORFC_DIR=/data4/workspace/zlt/featcodec/coding/vq/v3.4
PY=/home/user/anaconda3/envs/featcodec2/bin/python
RES=$ROOT/results/dinov2_vitl14/phase1/v33
SWEEP=$RES/sweep_20260808
LOGS=$SWEEP/logs
QUEUE=$SWEEP/queue.txt
LOCK=$SWEEP/queue.lock
PROGRESS=$SWEEP/progress.log

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

note () { printf '[%s][gpu%s] %s\n' "$(date +%H:%M:%S)" "${GPU:-?}" "$*" \
          | tee -a "$PROGRESS"; }

# --- queue primitives --------------------------------------------------------
pop_job () {
    local job=""
    exec 9>"$LOCK"; flock 9
    job=$(head -n 1 "$QUEUE" 2>/dev/null || true)
    [[ -n "$job" ]] && { tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"; }
    flock -u 9; exec 9>&-
    printf '%s' "$job"
}

push_front () {
    exec 9>"$LOCK"; flock 9
    { printf '%s\n' "$1"; cat "$QUEUE"; } > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE"
    flock -u 9; exec 9>&-
}

# --- job bodies --------------------------------------------------------------
# orfc:LAYER:K:EPOCHS  -- lambda=0 baseline in the ORFC codebase.
run_orfc () {
    local layer=$1 k=$2 epochs=$3
    local tag=orfc_${layer}_K${k}_ep${epochs}_lmbda0
    local lr_tag=0.0003
    local pt=$ORFC_DIR/checkpoints/dinov2_vitl14/${layer}_K${k}_emb32_bt1024_ws_tau0.5_lr${lr_tag}_ep${epochs}_n5000_s42.pt
    if [[ -f $pt ]]; then note "skip $tag (checkpoint exists)"; return 0; fi
    note "start $tag"
    cd "$ORFC_DIR" || return 1
    "$PY" -u run_soft_pq.py \
        --layer "$layer" --K "$k" --epochs "$epochs" \
        --embedding_dim 32 --bottleneck_dim 1024 --warm_start_opq \
        --lmbda 0.0 --tau_start 0.5 --tau_end 0.005 \
        --lr 3e-4 --max_train_images 5000 --seed 42 \
        2>&1 | tee "$LOGS/$tag.log"
    note "done  $tag"
}

# p2:BLK:RATE:ARM:SOURCE_DIR:ALLOCATION  -- 150-epoch specialised delivery.
run_p2 () {
    local blk=$1 rate=$2 arm=$3 src=$4 alloc=$5
    local tag=v33_ep150_${blk}_${rate}_${arm}
    if [[ -f $RES/$tag/$rate/checkpoint.pt ]]; then
        note "skip $tag (checkpoint exists)"; return 0
    fi
    note "start $tag"
    cd "$ROOT" || return 1
    "$PY" -u -m phase1.v33.train \
        --phase 2 --block "$blk" --anchor "$rate" --device cuda --run-id "$tag" \
        --source "$src" --allocation "$alloc" \
        --epochs 150 --batch 32 --ckpt-every 1000 --no-freeze-u0 --no-L \
        2>&1 | tee "$LOGS/$tag.log"
    note "done  $tag"
}

# pipe:BLK:RATE  -- phase 1 -> search -> searched arm, uniform arm re-queued.
run_pipe () {
    local blk=$1 rate=$2
    local run=v33_sweep_${blk}_${rate}
    local out=$RES/$run/$rate
    cd "$ROOT" || return 1

    if [[ ! -f $out/checkpoint.pt ]]; then
        note "start ${run} phase1"
        "$PY" -u -m phase1.v33.train \
            --phase 1 --block "$blk" --anchor "$rate" --device cuda \
            --run-id "$run" --epochs 50 --ckpt-every 500 \
            2>&1 | tee "$LOGS/${run}_phase1.log"
    fi
    [[ -f $out/checkpoint.pt ]] || { note "FAIL ${run} phase1"; return 1; }

    # Nothing in the tree materialises the uniform control allocation.
    if [[ ! -f $out/uniform_allocation.npy ]]; then
        "$PY" - "$blk" "$rate" "$out/uniform_allocation.npy" <<'PY'
import sys
import numpy as np
from phase1.v21.config import activate
from phase1.v12 import config as C
from phase1 import engine
activate(sys.argv[1])
anchor = C.ANCHOR_BY_NAME[sys.argv[2]]
np.save(sys.argv[3],
        np.asarray(engine.uniform_allocation(anchor, C.GROUPS), dtype=np.int64))
PY
    fi

    if [[ ! -f $out/allocation.npy ]]; then
        note "start ${run} search"
        "$PY" -u -m phase1.v33.search \
            --block "$blk" --anchor "$rate" --device cuda \
            --checkpoint "$out/checkpoint.pt" \
            --body-images 128 --recheck-images 500 --eval-budget 5000 \
            --propose-top 50 --out "$out/search.json" \
            2>&1 | tee "$LOGS/${run}_search.log"
    fi
    [[ -f $out/allocation.npy ]] || { note "FAIL ${run} search"; return 1; }

    # Hand the uniform control to whichever card frees up first.
    push_front "p2:$blk:$rate:uniform:$out:$out/uniform_allocation.npy"
    run_p2 "$blk" "$rate" searched "$out" "$out/allocation.npy"
}

# --- entry points ------------------------------------------------------------
case "${1:?usage: run_sweep_ep150.sh init | worker GPU}" in
init)
    mkdir -p "$LOGS"
    B05_R64=$RES/v33_blk05_R64_20260807T111820Z/R64
    B05_R96=$RES/v33_blk05_R96_20260807T112312Z/R96
    B20_R64=$RES/v33_formal_20260807T050000Z/R64
    B20_R96=$RES/v33_blk20_R96_20260807T111953Z/R96
    # blk20 R64 predates uniform_allocation.npy; its 75-epoch uniform arm has it.
    B20_R64_U=$RES/v33_noL_uniform_20260807T094204Z/R64/allocation.npy
    cat > "$QUEUE" <<EOF
orfc:blk05:8:300
pipe:blk10:R64
pipe:blk10:R96
pipe:blk15:R64
pipe:blk15:R96
p2:blk05:R64:searched:$B05_R64:$B05_R64/search_allocation.npy
p2:blk05:R64:uniform:$B05_R64:$B05_R64/uniform_allocation.npy
p2:blk05:R96:searched:$B05_R96:$B05_R96/search_allocation.npy
p2:blk05:R96:uniform:$B05_R96:$B05_R96/uniform_allocation.npy
orfc:blk10:8:100
orfc:blk15:8:100
orfc:blk10:4:100
orfc:blk15:4:100
orfc:blk20:8:100
p2:blk20:R64:searched:$B20_R64:$B20_R64/search_allocation.npy
p2:blk20:R64:uniform:$B20_R64:$B20_R64_U
p2:blk20:R96:searched:$B20_R96:$B20_R96/search_allocation.npy
p2:blk20:R96:uniform:$B20_R96:$B20_R96/uniform_allocation.npy
EOF
    printf 'queued %d jobs in %s\n' "$(wc -l < "$QUEUE")" "$QUEUE"
    ;;
worker)
    GPU=${2:?worker needs a GPU index}
    note "worker up"
    while :; do
        job=$(pop_job)
        [[ -z "$job" ]] && break
        IFS=: read -r kind a b c d e <<< "$job"
        case "$kind" in
        orfc) run_orfc "$a" "$b" "$c" ;;
        p2)   run_p2 "$a" "$b" "$c" "$d" "$e" ;;
        pipe) run_pipe "$a" "$b" ;;
        *)    note "unknown job $job" ;;
        esac
    done
    note "worker done (queue empty)"
    ;;
*)
    echo "usage: run_sweep_ep150.sh init | worker GPU" >&2
    exit 1
    ;;
esac
