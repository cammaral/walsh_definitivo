#!/usr/bin/env bash
# Prepare external dependencies without editing scientific algorithms.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v make >/dev/null || { echo "make is required" >&2; exit 1; }
command -v g++ >/dev/null || { echo "g++ is required (install Xcode Command Line Tools on macOS)" >&2; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -f .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install -r environment/requirements-reproduction.txt
else
  echo 'Existing .venv detected; leaving installed packages untouched.'
  .venv/bin/python -m pip check
fi

MQLIB_ROOT="${MQLIB_ROOT:-$ROOT/.external/MQLib}"
if [[ ! -d "$MQLIB_ROOT/.git" ]]; then
  if [[ -e "$MQLIB_ROOT" ]]; then
    echo "Cannot clone: $MQLIB_ROOT already exists but is not a git checkout" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$MQLIB_ROOT")"
  git clone https://github.com/MQLib/MQLib.git "$MQLIB_ROOT"
fi

# An exact revision can be supplied with MQLIB_COMMIT; otherwise capture the checkout revision.
if [[ -n "${MQLIB_COMMIT:-}" ]]; then
  git -C "$MQLIB_ROOT" fetch origin "$MQLIB_COMMIT"
  git -C "$MQLIB_ROOT" checkout --detach "$MQLIB_COMMIT"
fi

if [[ ! -x "$MQLIB_ROOT/bin/MQLib" || -n "${MQLIB_COMMIT:-}" ]]; then
  make -C "$MQLIB_ROOT"
fi
[[ -x "$MQLIB_ROOT/bin/MQLib" ]] || { echo "MQLib build failed" >&2; exit 1; }
mkdir -p results/reproducibility
REV="$(git -C "$MQLIB_ROOT" rev-parse HEAD)"
printf '%s\n' "$REV" > results/reproducibility/mqlib_commit.txt
printf 'MQLib ready at %s (commit %s)\n' "$MQLIB_ROOT" "$REV"
printf 'For this shell, use: export MQLIB_ROOT="%s"\n' "$MQLIB_ROOT"
