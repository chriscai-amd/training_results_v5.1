#!/usr/bin/env bash
# Process a list of Criteo days inside a docker container.
# Usage: bash process_days.sh "1 5 9 13 17 21" /criteo
set -eu
DAYS="$1"
ROOT="${2:-/criteo}"
mkdir -p "$ROOT/npy"
for d in $DAYS; do
    f="$ROOT/day_$d.gz"
    if [ ! -f "$f" ]; then
        echo "[skip] $f missing"
        continue
    fi
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] starting day_$d"
    python3 "$ROOT/preprocess_criteo_to_npy_gz.py" \
        --in-file "$f" --out-dir "$ROOT/npy" 2>&1 | sed "s/^/[d$d] /"
    echo "[$(date +%H:%M:%S)] finished day_$d"
done
echo "[$(date +%H:%M:%S)] ALL DONE for: $DAYS"
