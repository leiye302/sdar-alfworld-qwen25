#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
PYTHON="${SDAR_PYTHON:-python}"
mkdir -p runs
# Each size uses all eight allocated GPUs, sequentially, at the original global batch.
for size in 1.5b 3b; do
    "$PYTHON" src/run.py train --size "$size" 2>&1 | tee "runs/sdar_${size}.log"
    "$PYTHON" src/run.py export --size "$size" 2>&1 | tee "runs/export_${size}.log"
done
"$PYTHON" src/report.py
