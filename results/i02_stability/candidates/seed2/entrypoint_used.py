"""Bounded, single-seed reproduction of the historical seven-band candidate.

python scripts/i02_generate_candidate.py --seed2

No legacy spectral filter is called. A successful I01 certificate certifies the
saved field and full residual only; independent I02 stability checks are required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
I02 = ROOT / "results" / "i02_stability"
CODE = ("scripts/i02_generate_candidate.py", "scripts/ai_generation.py",
        "turing_rd/ai_init.py", "turing_rd/steady.py", "turing_rd/solver.py",
        "turing_rd/geometry.py", "turing_rd/mesh.py", "turing_rd/result_io.py",
        "turing_rd/diagnostics.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
        f.write("\n")


def worker(args):
    start, cpu = time.monotonic(), time.process_time()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(args.out / "runtime_cache" / "matplotlib")
    sys.path.insert(0, str(ROOT))
    import joblib
    import numpy as np
    from turing_rd import Geometry, STRIPES, icosphere
    from turing_rd.diagnostics import pattern_stats, full_steady_residual
    from turing_rd.result_io import save_certified_mat, read_mat_fields, verify_mat
    from turing_rd.steady import relax_polish, rotation_generators
    from scripts.ai_generation import ai_initial_state

    model_path = ROOT / "results" / "ai_generation" / "sphere5" / "model.joblib"
    historical = ROOT / "results" / "ai_generation" / "tasks" / "robustness_jobs" / "ai_sphere6_0.0540_0.0630_2.pkl"
    provenance = {
        "entrypoint": "scripts/i02_generate_candidate.py --seed2",
        "method": "historical_AI_initial_state_then_original_relax_polish",
        "seed": 2, "phase_radians": float(np.random.default_rng(2).uniform(0, np.pi)),
        "parameters": STRIPES.as_dict(), "surface": "sphere6", "laplacian": "robust",
        "robust_mollify_factor": 1e-5, "schedule": [[1.0, 2000.0], [4.0, 200000.0]],
        "check_every": 500.0, "tol": 1e-11, "orient": True, "threads": 1,
        "hard_timeout_seconds": args.timeout,
        "model_path": str(model_path.relative_to(ROOT)), "model_sha256": sha(model_path),
        "historical_diagnostic_cache": str(historical.relative_to(ROOT)),
        "historical_diagnostic_cache_sha256": sha(historical),
        "historical_cache_contains_fields": False,
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "scipy", "torch", "robust-laplacian", "scikit-learn", "joblib")},
        "python": sys.version,
        "model_environment_caveat": "Model stores sklearn 1.9.1 estimators; current runtime version recorded above. No model is modified or retrained.",
        "stability_scope": "not_evaluated_by_candidate_generator",
    }
    write_json(args.out / "provenance.json", provenance)
    print("Building sphere6 robust geometry and loading unchanged historical model", flush=True)
    geom = Geometry(icosphere(6), laplacian="robust", cache_dir=None)
    model = joblib.load(model_path)
    u0, v0, predicted = ai_initial_state(geom, model, STRIPES.F, STRIPES.k, 2)
    if predicted["type"] != "stripes_or_maze":
        raise ValueError("Historical model did not predict stripes_or_maze; stop rather than change initialisation")
    if u0.dtype != np.float64 or v0.dtype != np.float64:
        raise ValueError("Original initial fields must be float64")
    np.savez_compressed(args.out / "initial.npz", u=u0, v=v0)
    write_json(args.out / "initial.json", {"prediction": predicted, "phase_radians": provenance["phase_radians"],
               "dtype": "float64", "shape": pattern_stats(geom, v0),
               "full_residual": full_steady_residual(geom, STRIPES, u0, v0)})
    T = rotation_generators(geom)
    print("Starting single seed2; physical time cap=200000; parent hard timeout active", flush=True)
    solve_start, solve_cpu = time.monotonic(), time.process_time()
    u, v, info = relax_polish(geom, STRIPES, u0, v0,
                             schedule=((1.0, 2000.0), (4.0, 200000.0)),
                             check_every=500.0, tol=1e-11, rotations=T, orient=True, verbose=True)
    cost = {"solve_wall_seconds": time.monotonic() - solve_start,
            "solve_cpu_seconds": time.process_time() - solve_cpu}
    write_json(args.out / "solver_info.json", info)
    provenance["solver_cost"] = cost
    provenance["actual_relax_time"] = info["relax_time"]
    provenance["actual_relax_steps"] = info["relax_steps"]
    certificate = save_certified_mat(args.out / "solution.mat", geom, STRIPES, u, v,
                                     seed=2, tol=1e-11, provenance=provenance)
    # Re-read the actual MAT and independently repeat verification; no in-memory-only acceptance.
    ur, vr = read_mat_fields(args.out / "solution.mat", geom.n)
    if not np.array_equal(ur, u) or not np.array_equal(vr, v):
        raise RuntimeError("Saved MAT values differ from the solver's float64 fields")
    repeated = verify_mat(args.out / "solution.mat", geom, STRIPES, seed=2, tol=1e-11)
    if repeated["mat_sha256"] != certificate["mat_sha256"]:
        raise RuntimeError("MAT changed during verification")
    np.savez_compressed(args.out / "final.npz", u=ur, v=vr,
                        mat_sha256=certificate["mat_sha256"], dtype="float64")
    shape = pattern_stats(geom, vr)
    accepted = bool(certificate["accepted"] and repeated["accepted"] and
                    shape.get("type") == "stripes_or_maze" and shape.get("n_components") == 7)
    cost.update(total_worker_wall_seconds=time.monotonic() - start,
                total_worker_cpu_seconds=time.process_time() - cpu)
    summary = {"pipeline": "i02_historical_seed2_reproduction", "surface": "sphere6", "seed": 2,
               **STRIPES.as_dict(), "mat_certificate": certificate,
               "mat_readback_accepted": certificate["accepted"],
               "rel_residual": certificate["residual"]["value"],
               "pattern": shape, "candidate_generation_accepted": accepted,
               "solver_info": info, "cost": cost, "provenance": provenance,
               "full_discrete_stability": "not_yet_evaluated",
               "note": "Generation acceptance is I01 full-residual/structure/precision plus seven-band shape; no stability assertion."}
    write_json(args.out / "summary.json", summary)
    print(json.dumps({"candidate_generation_accepted": accepted, "residual": summary["rel_residual"],
                      "n_components": shape.get("n_components"), "cost": cost}), flush=True)
    return 0 if accepted else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed2", action="store_true", required=True, help="Run only historical candidate seed2")
    ap.add_argument("--timeout", type=float, default=1800.0, help="Hard worker wall cap, 0 < seconds <= 1800")
    ap.add_argument("--out", type=Path, default=I02 / "candidates" / "seed2")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out = args.out.resolve()
    if not 0 < args.timeout <= 1800:
        ap.error("Timeout must be positive and at most 1800 seconds")
    if not args.out.is_relative_to(I02.resolve()) or args.out == I02.resolve():
        ap.error("All new outputs must be in a new child directory of results/i02_stability")
    if args.worker:
        return worker(args)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = {"seed": 2, "timeout_seconds": args.timeout, "accepted": False,
              "stability_status": "not_evaluated_by_generator", "timed_out": False}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    try:
        with (args.out / "generation.log").open("x", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--seed2",
                                   "--worker", "--out", str(args.out), "--timeout", str(args.timeout)],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=args.timeout, check=False)
        status.update(returncode=proc.returncode, accepted=proc.returncode == 0)
        if proc.returncode:
            status["failure_reason"] = "Worker failed or full-residual/float64/seven-band acceptance failed; inspect generation.log and any saved candidate."
    except subprocess.TimeoutExpired:
        status.update(timed_out=True, failure_reason="Hard wall cap reached; worker terminated. No pass is certified.")
    except Exception as exc:
        status["failure_reason"] = repr(exc)
    status["parent_wall_seconds"] = time.monotonic() - start
    status["command"] = [sys.executable, "scripts/i02_generate_candidate.py", "--seed2",
                         "--out", str(args.out), "--timeout", str(args.timeout)]
    write_json(args.out / "run_manifest.json", status)
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
