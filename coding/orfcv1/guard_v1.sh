#!/bin/bash
# Read-only half-hour guard for one formal ORFC-v1 run.

set -u

if [ "$#" -ne 1 ]; then
    echo "usage: $0 RUN_ID" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ID="$1"
LOG_DIR="$SCRIPT_DIR/logs/$RUN_ID"
RESULTS_DIR="$SCRIPT_DIR/results/dinov2_vitl14/$RUN_ID"
GUARD_LOG="$LOG_DIR/guard.log"
ALERT_FILE="$LOG_DIR/GUARD_ALERT"
PYTHON="/home/user/anaconda3/envs/featcodec2/bin/python"
INTERVAL_SECONDS="${GUARD_INTERVAL_SECONDS:-1800}"
mkdir -p "$LOG_DIR"

snapshot() {
    {
        echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_id=$RUN_ID ====="
        echo "[tmux]"
        tmux list-panes -t zlt1 \
            -F '#{session_name}:#{window_index}.#{pane_index} cmd=#{pane_current_command} dead=#{pane_dead} pid=#{pane_pid}' \
            2>&1 || true
        echo "[gpu]"
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader 2>&1 || true
        echo "[process]"
        pgrep -a -u "$USER" -f \
            'run_exp_v1.sh|run_v1.py|select_phasec.py|select_final.py' \
            2>&1 || true
        echo "[log-errors]"
        if [ -d "$LOG_DIR" ]; then
            rg -n -i \
                'Traceback|CUDA out of memory|FATAL:| ERROR:|(^|[^A-Za-z])NaN([^A-Za-z]|$)|(^|[^A-Za-z])Inf([^A-Za-z]|$)' \
                "$LOG_DIR" --glob '*.log' 2>&1 | tail -40 || true
        fi
        echo "[result-contracts]"
        "$PYTHON" - "$RESULTS_DIR" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
paths = sorted(glob.glob(os.path.join(root, '*.json')))
rows = []
violations = []
for path in paths:
    name = os.path.basename(path)
    if name.endswith('_selection.json') or name in {
            'phasec_selection.json', 'final_selection.json',
            'chunk_benchmark_report.json'}:
        continue
    try:
        with open(path) as f:
            r = json.load(f)
    except Exception as exc:
        violations.append(f'{name}: unreadable JSON: {exc}')
        continue
    cfg = r.get('config', {})
    epochs = int(cfg.get('epochs', -1))
    history = r.get('history', [])
    audit = r.get('reconstruction_audit', {})
    status = 'complete' if len(history) == epochs and audit.get('passed') else 'partial'
    rows.append((name, status, len(history), epochs))
    if audit and not audit.get('passed', False):
        violations.append(f'{name}: reconstruction audit failed')
    val = r.get('heldout_metrics', {}).get('val')
    if val:
        n = val.get('eps_g', {}).get('n_images')
        for key in ('eps_g', 'q_g_normalized', 'kappa_g'):
            groups = val.get(key, {}).get('per_group', {})
            bad = [g for g, x in groups.items() if x.get('n') != n]
            if bad:
                violations.append(f'{name}: {key} count mismatch groups={bad[:4]}')
        if not val.get('interaction', {}).get('per_pair'):
            violations.append(f'{name}: validation interaction missing')
print(f'json_files={len(paths)} experiment_rows={len(rows)}')
for row in rows[-12:]:
    print(f'{row[1]:8s} history={row[2]}/{row[3]} {row[0]}')
if violations:
    print('VIOLATIONS:')
    for item in violations:
        print(item)
    raise SystemExit(3)
PY
        audit_exit=$?
        echo "contract_audit_exit=$audit_exit"
        echo
        return "$audit_exit"
    } >> "$GUARD_LOG" 2>&1
}

while true; do
    snapshot
    status=$?
    if [ "$status" -ne 0 ]; then
        printf '%s contract guard failed with exit %s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" > "$ALERT_FILE"
    fi

    if rg -q "ORFC-v1 delivery run completed: $RUN_ID" \
            "$LOG_DIR/driver.log" 2>/dev/null; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) completed" >> "$GUARD_LOG"
        exit "$status"
    fi

    if ! pgrep -u "$USER" -f '[r]un_exp_v1.sh' >/dev/null 2>&1 \
            && ! pgrep -u "$USER" -f '[r]un_v1.py' >/dev/null 2>&1; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) run stopped before completion" \
            >> "$GUARD_LOG"
        printf '%s run stopped before completion\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ALERT_FILE"
        exit 4
    fi
    sleep "$INTERVAL_SECONDS"
done
