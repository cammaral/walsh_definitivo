#!/usr/bin/env bash
# Stage 06 requires the Stage 05 .npz vectors and selected_best_raw.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
MODE="${1:---help}"
mkdir -p results/reproducibility
case "$MODE" in
  --self-test) "$PY" stages/06_transfer/run.py --self-test ;;
  --dry-run) "$PY" stages/06_transfer/run.py --dry-run ;;
  --statevector) "$PY" stages/06_transfer/run.py --statevector-only 2>&1 | tee results/reproducibility/06_statevector.log; "$PY" scripts/verify_paper_results.py --through 06 ;;
  --full) "$PY" stages/06_transfer/run.py 2>&1 | tee results/reproducibility/06_full.log; "$PY" scripts/verify_paper_results.py --through 06 ;;
  *) echo "Usage: bash scripts/reproduce_transfer.sh [--self-test|--dry-run|--statevector|--full]" ;;
esac
