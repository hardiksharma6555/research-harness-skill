#!/usr/bin/env bash
# Extract CLIP features for every frozen split matching the given globs (default: sd14 train/val + all test sets).
# Naming: frozen/splits/train/sd14_train.tsv -> features/clip_l14/sd14_train ; frozen/splits/test/adm.tsv -> features/clip_l14/test_adm
# Usage (through launch.py): bash src/extract_all.sh [GLOB ...]
set -euo pipefail
cd "$(dirname "$0")/.."
globs=("$@")
[ ${#globs[@]} -eq 0 ] && globs=("frozen/splits/train/sd14_*.tsv" "frozen/splits/test/*.tsv")
for g in "${globs[@]}"; do
  for f in $g; do
    b=$(basename "$f" .tsv)
    case "$f" in */test/*) out="features/clip_l14/test_$b" ;; *) out="features/clip_l14/$b" ;; esac
    "$PY" src/extract_features.py --split "$f" --out "$out"
  done
done
echo "EXTRACT_ALL done"
