"""Run the actual unit suite and save version-bound counts for report generation."""
import argparse
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evidence_stats import source_fingerprint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("test evidence exists; use a fresh output")
    before = source_fingerprint(ROOT)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {"schema_version": "executed-unittest-v1", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "successful": result.wasSuccessful(),
              "source_fingerprint": before, "sources_unchanged": before == source_fingerprint(ROOT),
              "wall_seconds": time.perf_counter() - started}
    if not record["sources_unchanged"]:
        record["successful"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    return 0 if record["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
