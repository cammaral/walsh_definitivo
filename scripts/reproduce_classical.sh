#!/usr/bin/env bash
# Full original classical pipeline: stages 01–05. This is expensive.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ "${1:-}" != "--run" ]]; then
  echo "Usage: bash scripts/reproduce_classical.sh --run"
  echo "This executes 17,500 MQLib heuristic runs and many thousands of optimizer runs."
  echo "Read REPRODUCIBILITY.md first. The project should be run in a fresh copy."
  exit 0
fi
export MQLIB_ROOT="${MQLIB_ROOT:-$ROOT/.external/MQLib}"
[[ -x "$MQLIB_ROOT/bin/MQLib" ]] || { echo "Missing MQLib executable: $MQLIB_ROOT/bin/MQLib"; exit 1; }
[[ -x "$PY" ]] || { echo "Missing Python executable: $PY"; exit 1; }
mkdir -p results/reproducibility
"$PY" scripts/capture_environment.py
"$PY" scripts/check_input_hashes.py
run() { local label="$1"; shift; echo "Starting: $label"; "$PY" "$@" 2>&1 | tee "results/reproducibility/${label}.log"; }
# Graph generation overwrites graph .txt files. Verify exact original hashes afterwards.
run 01_generate stages/01_random_graphs/generate_graphs.py
"$PY" scripts/check_input_hashes.py
run 01_mqlib stages/01_random_graphs/run_mqlib.py
run 02_split stages/02_calibration/create_split.py
run 02_alpha stages/02_calibration/run_alpha.py
run 02_beta stages/02_calibration/run_beta.py
run 03_complete stages/03_complete_basis/run_complete_basis.py
run 04_grid stages/04_support_sweep/prepare_support_grid.py
run 04_sweep stages/04_support_sweep/run_support_sweep.py
run 05_gset_B1 stages/05_gset/run_gset.py
"$PY" scripts/verify_paper_results.py --through 05
echo "Classical pipeline completed. Inspect verification report."
