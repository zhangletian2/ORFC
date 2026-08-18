#!/usr/bin/env bash
set -euo pipefail

cd /data4/workspace/zlt/featcodec/ORFC/coding/orfcv1
run_id="${1:-block_codec_actual_rans_20260818}"
IFS=',' read -r -a gpus <<< "${GPUS:-0,1,2,3}"
result_root="results/dinov2_vitl14/phase1/v34/block_codec_benchmark/${run_id}"
log_root="logs/${run_id}"
mkdir -p "${result_root}" "${log_root}"

jobs=()
for block in blk05 blk10 blk15 blk20; do
  jobs+=("${block}:orfc:results/dinov2_vitl14/phase1/v34/block_vq_4depth_100ep_20260815/${block}/orfc")
  jobs+=("${block}:block_k256:results/dinov2_vitl14/phase1/v34/block_vq_4depth_100ep_20260815/${block}/block")
  jobs+=("${block}:block_pq:results/dinov2_vitl14/phase1/v34/block_pq_e16k2_warmlloyd_4depth_100ep_20260817/${block}/block_pq")
done

pids=()
names=()
wait_wave() {
  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "FAILED ${names[$i]}" >&2
      failed=1
    fi
  done
  pids=()
  names=()
  if (( failed )); then exit 1; fi
}

for item in "${jobs[@]}"; do
  IFS=':' read -r block arm path <<< "${item}"
  index=${#pids[@]}
  gpu=${gpus[$index]}
  output="${result_root}/${block}/${arm}.json"
  if [[ -e "${output}" ]]; then
    echo "refusing to overwrite ${output}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${output}")"
  echo "START ${block}/${arm} GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" /home/user/anaconda3/envs/featcodec2/bin/python \
    -m phase1.v34.block_codec_benchmark \
    --checkpoint "${path}/codec.pt" --downstream "${path}/downstream.json" \
    --block "${block}" --train-images 5000 --cls-images 500 --voc-images 100 \
    --batch 32 --context-buckets 4096 --alpha 0.5 \
    --timing-batches 1,32 --timing-repeats 20 --device cuda:0 \
    --output "${output}" > "${log_root}/${block}_${arm}.log" 2>&1 &
  pids+=("$!")
  names+=("${block}/${arm}")
  if (( ${#pids[@]} == ${#gpus[@]} )); then wait_wave; fi
done
if (( ${#pids[@]} )); then wait_wave; fi
echo "COMPLETE ${result_root}"
