"""Verify that the I01/I02 baseline files are still byte-identical (read-only check).

    python scripts/i04_protected_check.py

Compares every entry of results/i02_stability/baseline_manifest.json with the file on disk.
Never writes to the protected files.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "i02_stability" / "baseline_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    protected = data["protected_files"]
    missing, changed = [], []
    for relative, expected in protected.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = sha256(path)
        if actual != expected:
            changed.append(relative)
    print(f"checked_at_utc: {datetime.now(timezone.utc).isoformat()}")
    print(f"baseline_head: {data.get('head')}")
    print(f"branch: {data.get('branch')}")
    print(f"protected_files: {len(protected)}  checked: {len(protected) - len(missing)}")
    print(f"missing: {len(missing)}  changed: {len(changed)}")
    for relative in missing[:20]:
        print(f"MISSING {relative}")
    for relative in changed[:20]:
        print(f"CHANGED {relative}")
    print("RESULT: PROTECTED_FILES_UNCHANGED" if not missing and not changed
          else "RESULT: PROTECTED_FILES_CHANGED")
    return 0 if not missing and not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
