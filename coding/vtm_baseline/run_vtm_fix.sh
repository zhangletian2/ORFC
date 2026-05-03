#!/bin/bash
# 修复脚本：合并 test 的 500 条 stats 到 test2，并重跑缺失的 blk05 QP=22,25
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FEAT_ROOT=$PROJECT_ROOT/features/test2
TEST_ROOT=$PROJECT_ROOT/features/test
BIT_DEPTH=10
VTM_ENCODER=$SCRIPT_DIR/EncoderAppStatic
VTM_DECODER=$SCRIPT_DIR/DecoderAppStatic
VTM_CFG=$SCRIPT_DIR/encoder_intra_vtm.cfg
TMP_DIR=$SCRIPT_DIR/_vtm_tmp
WORKERS=8

echo "=========================================="
echo "Step 1: 合并 test stats → test2 CSV，并清理无 stats 的 decoded npy"
echo "=========================================="

python3 - "$TEST_ROOT" "$FEAT_ROOT" <<'PYEOF'
import csv, os, sys
from collections import defaultdict
from pathlib import Path

test_root = Path(sys.argv[1]) / "dinov2_vitl14" / "decoded" / "vtm"
test2_root = Path(sys.argv[2]) / "dinov2_vitl14" / "decoded" / "vtm"

BLK_QP = {
    "blk05": [22, 25, 27, 30, 32, 35],
    "blk10": [22, 25, 27, 30, 32, 35],
    "blk15": [22, 25, 27, 30, 32, 35],
    "blk20": [0, 2, 5, 7, 10, 12],
}

# ---- 读取 test stats ----
test_data = {}  # (layer, qp, filename) -> row
for qp_dir in os.listdir(test_root):
    csv_path = test_root / qp_dir / "_stats.csv"
    if not csv_path.is_file():
        continue
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            test_data[(row[2], str(row[3]), row[0])] = row

print(f"[INFO] 从 test 读取了 {len(test_data)} 条 stats")

# ---- 处理每个 QP 的 CSV ----
csv_header = ["filename","model","layer","qp","bitstream_bytes","bits","bpfp",
              "encode_s","decode_s","total_s"]

merged_total = 0
deleted_total = 0

for layer, qps in BLK_QP.items():
    for qp in qps:
        csv_path = test2_root / str(qp) / "_stats.csv"
        npy_dir = test2_root / str(qp) / layer

        # 读取 test2 已有 stats
        existing = {}  # filename -> row (只看本 layer)
        other_rows = []
        if csv_path.is_file():
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) < 4:
                        continue
                    if row[2] == layer and str(row[3]) == str(qp):
                        existing[row[0]] = row
                    else:
                        other_rows.append(row)

        # 获取 test2 中本 layer 的所有 decoded npy
        npy_names = set()
        if npy_dir.is_dir():
            npy_names = {f[:-4] for f in os.listdir(npy_dir) if f.endswith(".npy")}

        # 合并 test stats（仅限 test2 中存在 npy 的文件）
        added = 0
        for fname in sorted(npy_names):
            if fname not in existing:
                key = (layer, str(qp), fname)
                if key in test_data:
                    existing[fname] = test_data[key]
                    added += 1

        if added > 0:
            merged_total += added
            print(f"  [MERGE] {layer}/QP{qp}: +{added} from test → {len(existing)} total")

        # 找出 有 npy 但没有 stats 的文件 → 删除 npy 以便重跑
        no_stats = npy_names - set(existing.keys())
        if no_stats:
            for fname in no_stats:
                npy_path = npy_dir / f"{fname}.npy"
                npy_path.unlink()
            deleted_total += len(no_stats)
            print(f"  [CLEAN] {layer}/QP{qp}: deleted {len(no_stats)} decoded npy (no stats)")

        # 重写 CSV
        all_rows = sorted(existing.values(), key=lambda r: (r[2], r[0]))
        all_rows = other_rows + all_rows
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            w.writerows(all_rows)

print(f"\n[DONE] 合并了 {merged_total} 条 stats，删除了 {deleted_total} 个无 stats 的 npy")
PYEOF

echo ""
echo "=========================================="
echo "Step 2: 重跑缺失的 blk05 QP=22,25"
echo "=========================================="

python vtm_baseline.py \
  --feat_root $FEAT_ROOT \
  --models dinov2_vitl14 \
  --layers blk05 \
  --qps 22 25 \
  --bit_depth $BIT_DEPTH \
  --vtm_encoder $VTM_ENCODER \
  --vtm_decoder $VTM_DECODER \
  --vtm_cfg $VTM_CFG \
  --tmp_dir $TMP_DIR \
  --workers $WORKERS

echo ""
echo "=========================================="
echo "Step 3: 验证结果完整性"
echo "=========================================="

python3 - "$FEAT_ROOT" <<'PYEOF'
import csv, os, sys
from pathlib import Path

feat_root = Path(sys.argv[1]) / "dinov2_vitl14" / "decoded" / "vtm"

BLK_QP = {
    "blk05": [22, 25, 27, 30, 32, 35],
    "blk10": [22, 25, 27, 30, 32, 35],
    "blk15": [22, 25, 27, 30, 32, 35],
    "blk20": [0, 2, 5, 7, 10, 12],
}

print(f"{'Block':<8} {'QP':<6} {'npy':<6} {'stats':<8} {'avg_bpfp':<12} {'status'}")
print("-" * 55)

all_ok = True
for layer in ["blk05", "blk10", "blk15", "blk20"]:
    for qp in BLK_QP[layer]:
        npy_dir = feat_root / str(qp) / layer
        npy_count = len([f for f in os.listdir(npy_dir) if f.endswith(".npy")]) if npy_dir.is_dir() else 0

        csv_path = feat_root / str(qp) / "_stats.csv"
        stats_count = 0
        bpfp_vals = []
        if csv_path.is_file():
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["layer"] == layer and str(row["qp"]) == str(qp):
                        stats_count += 1
                        bpfp_vals.append(float(row["bpfp"]))

        avg_bpfp = sum(bpfp_vals) / len(bpfp_vals) if bpfp_vals else 0
        ok = npy_count == 2000 and stats_count == 2000
        status = "OK" if ok else "INCOMPLETE"
        if not ok:
            all_ok = False
        print(f"{layer:<8} {qp:<6} {npy_count:<6} {stats_count:<8} {avg_bpfp:<12.6f} {status}")
    print()

if all_ok:
    print("==> ALL COMPLETE! 所有 block/QP 均为 2000 条完整数据")
else:
    print("==> WARNING: 部分数据不完整，请检查")
PYEOF

echo ""
echo "All done!"
