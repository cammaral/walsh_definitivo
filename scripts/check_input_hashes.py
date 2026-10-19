#!/usr/bin/env python3
"""Verify that graph input files match the bank bundled with the reviewed ZIP."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "input_sha256.tsv"

def main():
    errors = []
    lines = MANIFEST.read_text().splitlines()
    for line in lines:
        expected, relative = line.split("\t", 1)
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"MISSING: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"CHANGED: {relative}")
    if errors:
        print("\n".join(errors[:30]))
        raise SystemExit(f"Input hash verification FAILED: {len(errors)} of {len(lines)} files")
    print(f"Input hash verification OK: {len(lines)} files")

if __name__ == "__main__":
    main()
