#!/usr/bin/env python3
"""Capture execution metadata without modifying the computational pipeline."""
from pathlib import Path
import datetime as dt
import json
import os
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproducibility"

def run(*args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except OSError:
        return None

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pip = run(sys.executable, "-m", "pip", "freeze") or ""
    (OUT / "pip_freeze.txt").write_text(pip + "\n")
    mqlib_root = os.environ.get("MQLIB_ROOT")
    info = {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mqlib_root": mqlib_root,
        "mqlib_git_commit": run("git", "-C", mqlib_root, "rev-parse", "HEAD") if mqlib_root else None,
        "project_git_commit": run("git", "-C", str(ROOT), "rev-parse", "HEAD"),
    }
    (OUT / "environment.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Environment captured at {OUT}")

if __name__ == "__main__":
    main()
