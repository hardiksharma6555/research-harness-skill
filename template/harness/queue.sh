#!/usr/bin/env bash
# Run several launches one after another, waiting for each run's DONE file (shared filesystem, no ssh polling).
# Run it on the login node with nohup; watch QUEUE_OUT for the final "QUEUE_DONE" line.
#
# Usage: PY=/path/to/python bash harness/queue.sh QUEUE_FILE > QUEUE_OUT 2>&1
# QUEUE_FILE: one run per line:  RUN_ID TAG SETTING COMMAND...   (blank lines and # comments are skipped)
set -u
cd "$(dirname "$0")/.."
: "${PY:?set PY to the python interpreter}"
MAX_WAIT_S="${MAX_WAIT_S:-3600}"
while read -r id tag setting cmd; do
  case "$id" in ""|\#*) continue ;; esac
  # shellcheck disable=SC2086
  "$PY" harness/launch.py "$id" --tag "$tag" --setting "$setting" --gpu auto -- $cmd 2>&1 | grep -E "LAUNCHED|ERROR"
  if [ ! -d "runs/$id" ]; then echo "$id NOT_LAUNCHED"; continue; fi
  waited=0
  until [ -f "runs/$id/DONE" ] || [ "$waited" -ge "$MAX_WAIT_S" ]; do sleep 5; waited=$((waited + 5)); done
  echo "$id rc=$(cat "runs/$id/DONE" 2>/dev/null || echo TIMEOUT)"
  grep -E "^(TRAINED|EVAL|GATE)|Error" "runs/$id/log.txt" | cut -c1-240
done < "$1"
echo QUEUE_DONE
