"""Bounded independent spectra for an existing task-3 MAT; no PDE, training or replacement."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def worker(args):
    from turing_rd import Geometry, GSParams, load_frog
    from turing_rd.spectrum_evidence import save_evidence, validate_evidence, write_json
    from i03_frog_stability import solve_config
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    cert = summary["mat_certificate"]; p = GSParams(**cert["parameters"])
    g = Geometry(load_frog(args.mesh), laplacian="robust", cache_dir=None)
    if args.verify:
        result = validate_evidence(args.verify, args.mat, g, p, cert["seed"])
    else:
        print("read-only existing MAT -> three unprojected shift-invert solves", flush=True)
        evidence = save_evidence(args.mat, cert, g, p, args.out / "evidence", solve_config)
        if not evidence["decision"]["numerical_check_passed"]:
            return 1
        result = validate_evidence(args.out / "evidence/certificate.json", args.mat, g, p, cert["seed"])
        write_json(args.out / "verification.json", result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "submission/task3_inversion/best_match"
    ap.add_argument("--mat", type=Path, default=base / "solution_best_match.mat")
    ap.add_argument("--summary", type=Path, default=base / "summary.json")
    ap.add_argument("--mesh", type=Path, default=ROOT / "frog.mat")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", type=Path, help="read saved vectors and recompute their residuals, no fresh eigensolve")
    ap.add_argument("--seconds", type=float, default=180.)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker or args.verify:
        return worker(args)
    if args.out is None or args.out.exists() or not 0 < args.seconds <= 300:
        ap.error("provide a fresh --out and 0 < --seconds <= 300")
    args.out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--out", str(args.out),
               "--mat", str(args.mat), "--summary", str(args.summary), "--mesh", str(args.mesh)]
    try:
        with (args.out / "run.log").open("x", encoding="utf-8") as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds).returncode
    except subprocess.TimeoutExpired:
        from turing_rd.spectrum_evidence import write_json
        write_json(args.out / "timeout.json", {"status": "incomplete", "seconds": args.seconds,
                    "fresh_spectral_check_completed": False, "reason": "hard wall-clock limit"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
