#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
PASS=0
FAIL=0

check() {
    local name="$1"
    shift

    echo
    echo "========== $name =========="

    if "$@"; then
        echo "[PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

# 1. Ambiente Python
check "Python dependencies" "$PY" -m pip check

# 2. Integridade dos dados originais
check "Graph input hashes" \
    "$PY" scripts/check_input_hashes.py

# 3. MQLib compilado
check "MQLib executable" \
    test -x "$ROOT/.external/MQLib/bin/MQLib"

# 4. Importação dos módulos científicos
check "Walsh-PCE imports" "$PY" -c '
import sys
sys.path.insert(0, "src")

import walsh_pce.core
import walsh_pce.correlators
import walsh_pce.decode
import walsh_pce.graphs
import walsh_pce.initialization
import walsh_pce.io
import walsh_pce.local_search
import walsh_pce.loss
import walsh_pce.masks
import walsh_pce.protocol
import walsh_pce.seeds
import walsh_pce.sparse
import walsh_pce.support
import walsh_pce.training

print("All scientific modules imported successfully")
'

# 5. Sintaxe Python de todos os estágios
# compile() não executa os scripts nem cria __pycache__
check "Stages 01-07 syntax" "$PY" -c '
from pathlib import Path

files = sorted(Path("stages").rglob("*.py"))
for path in files:
    source = path.read_bytes()
    compile(source, str(path), "exec")

print(f"Syntax OK: {len(files)} scripts")
'

# 6. Teste funcional do Walsh loader
check "Stage 06 self-test" \
    "$PY" stages/06_transfer/run.py --self-test

# 7. Verificação dos recursos do Stage 06
check "Stage 06 dry-run" \
    "$PY" stages/06_transfer/run.py --dry-run

echo
echo "=================================="
echo "SMOKE TEST SUMMARY"
echo "PASS: $PASS"
echo "FAIL: $FAIL"
echo "=================================="

if [ "$FAIL" -eq 0 ]; then
    echo "All smoke tests passed."
else
    exit 1
fi
