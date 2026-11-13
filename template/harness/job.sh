#!/usr/bin/env bash
# Runs on the compute node. Started only by harness/launch.py; do not call directly.
# Runs the command, then (if the run wrote scores/) the frozen evaluator and the gate.
# Writes runs/<id>/DONE with the exit code last, so a watcher can wait on that one file.
set -u
ID="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RD="$ROOT/runs/$ID"
cd "$ROOT" || exit 1
export RUN_DIR="$RD"
start=$(date +%s)
"$PY" harness/trace.py running "$ID" gpu="${CUDA_VISIBLE_DEVICES:-none}"
bash "$RD/cmd.sh" > "$RD/log.txt" 2>&1
rc=$?
if [ "$rc" -eq 0 ] && [ -d "$RD/scores" ]; then
  "$PY" frozen/eval.py "$RD" >> "$RD/log.txt" 2>&1 || rc=97
  "$PY" harness/gate.py --from-run "$RD" >> "$RD/log.txt" 2>&1 || true
fi
"$PY" harness/trace.py done "$ID" rc="$rc" seconds="$(( $(date +%s) - start ))"
echo "$rc" > "$RD/DONE"
