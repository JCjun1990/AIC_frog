"""Bounded, single-seed reproduction of the historical seven-band candidate.

Two-stage use (the I02 continuation protocol):

    python scripts/i02_generate_candidate.py --seed2 --stage relax --out results/i02_stability/candidates/seed2_v2
    python scripts/i02_generate_candidate.py --seed2 --stage finish --out results/i02_stability/candidates/seed2_v2

``--stage relax`` runs the historical AI initial state and the original relax loop
with rotation-corrected convergence, then writes the float64 relative-equilibrium
checkpoint *before* ``orient_polish`` is ever entered.  ``--stage finish`` resumes
from that checkpoint, follows the relative-equilibrium family to a true steady
state, and saves/verifies the MAT.  ``--stage all`` runs both in one process.

No legacy spectral filter is called.  A successful I01 certificate certifies the
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
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:       # importable when loaded by path (tests) too
    sys.path.insert(0, str(HERE))
from json_safe import write_json_exclusive  # noqa: E402

I02 = ROOT / "results" / "i02_stability"
SCHEDULE = ((1.0, 2000.0), (4.0, 200000.0))
TOL = 1e-11
CHECKPOINT = "checkpoint_relative_equilibrium.npz"
CODE = ("scripts/i02_generate_candidate.py", "scripts/json_safe.py", "scripts/ai_generation.py",
        "turing_rd/ai_init.py", "turing_rd/steady.py", "turing_rd/solver.py",
        "turing_rd/geometry.py", "turing_rd/mesh.py", "turing_rd/result_io.py",
        "turing_rd/diagnostics.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    """Serialize the whole document before creating the file (never truncate evidence)."""
    return write_json_exclusive(path, data)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stop_worker_tree(proc):
    """Terminate only a live worker launched by this parent, including Windows venv children."""
    result = {"worker_launcher_pid": proc.pid, "already_exited": proc.poll() is not None}
    if result["already_exited"]:
        return result
    if os.name == "nt":
        try:
            cleanup = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                     capture_output=True, text=True, timeout=15, check=False)
            result.update(returncode=cleanup.returncode, stdout=cleanup.stdout, stderr=cleanup.stderr)
        except Exception as exc:
            result["tree_cleanup_error"] = repr(exc)
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=15)
    result["launcher_exited"] = proc.poll() is None
    return result


def geometry():
    from turing_rd import Geometry, icosphere
    return Geometry(icosphere(6), laplacian="robust", cache_dir=None)


def provenance(args, model_path, historical):
    import numpy as np
    from turing_rd import STRIPES
    return {
        "entrypoint": "scripts/i02_generate_candidate.py --seed2",
        "method": "historical_AI_initial_state_then_original_relax_polish",
        "stages": {"requested": args.stage,
                   "relax": "AI initial state + original relax loop, rotation-corrected residual",
                   "finish": "orient_polish along the relative-equilibrium family + newton_polish"},
        "seed": 2, "phase_radians": float(np.random.default_rng(2).uniform(0, np.pi)),
        "parameters": STRIPES.as_dict(), "surface": "sphere6", "laplacian": "robust",
        "robust_mollify_factor": 1e-5, "schedule": [list(x) for x in SCHEDULE],
        "check_every": 500.0, "tol": TOL, "orient": True, "threads": 1,
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


def relax_stage(args):
    """Historical initial state -> original relaxation -> float64 relative-equilibrium checkpoint."""
    import joblib
    import numpy as np
    from turing_rd import STRIPES
    from turing_rd.diagnostics import full_steady_residual, pattern_stats
    from turing_rd.steady import relax_polish, rotation_generators
    from scripts.ai_generation import ai_initial_state

    started, cpu = time.monotonic(), time.process_time()
    model_path = ROOT / "results" / "ai_generation" / "sphere5" / "model.joblib"
    historical = (ROOT / "results" / "ai_generation" / "tasks" / "robustness_jobs" /
                  "ai_sphere6_0.0540_0.0630_2.pkl")
    prov = provenance(args, model_path, historical)
    write_json(args.out / "provenance.json", prov)
    print("Building sphere6 robust geometry and loading unchanged historical model", flush=True)
    geom = geometry()
    model = joblib.load(model_path)
    u0, v0, predicted = ai_initial_state(geom, model, STRIPES.F, STRIPES.k, 2)
    if predicted["type"] != "stripes_or_maze":
        raise ValueError("Historical model did not predict stripes_or_maze; stop rather than change initialisation")
    if u0.dtype != np.float64 or v0.dtype != np.float64:
        raise ValueError("Original initial fields must be float64")
    np.savez_compressed(args.out / "initial.npz", u=u0, v=v0)
    write_json(args.out / "initial.json", {"prediction": predicted,
               "phase_radians": prov["phase_radians"], "dtype": "float64",
               "shape": pattern_stats(geom, v0),
               "full_residual": full_steady_residual(geom, STRIPES, u0, v0)})
    T = rotation_generators(geom)
    print(f"Starting single seed2 relaxation; physical time cap={SCHEDULE[-1][1]:.0f}; "
          f"parent stage timeout active", flush=True)
    solve_start, solve_cpu = time.monotonic(), time.process_time()
    u, v, info = relax_polish(geom, STRIPES, u0, v0, schedule=SCHEDULE, check_every=500.0,
                              tol=TOL, rotations=T, orient=False, verbose=True)
    cost = {"relax_wall_seconds": time.monotonic() - solve_start,
            "relax_cpu_seconds": time.process_time() - solve_cpu,
            "stage_wall_seconds": time.monotonic() - started,
            "stage_cpu_seconds": time.process_time() - cpu}
    checkpoint = args.out / CHECKPOINT
    np.savez_compressed(checkpoint, u=np.asarray(u, dtype=np.float64),
                        v=np.asarray(v, dtype=np.float64),
                        omega=np.asarray(info["omega"], dtype=np.float64),
                        rel_residual=np.float64(info["rel_residual"]),
                        rel_residual_rot=np.float64(info["rel_residual_rot"]))
    # The checkpoint must be a bit-exact float64 copy before later stages rely on it.
    with np.load(checkpoint) as saved:
        exact = bool(np.array_equal(saved["u"], u) and np.array_equal(saved["v"], v))
    if not exact:
        raise RuntimeError("Relative-equilibrium checkpoint is not a bit-exact float64 copy")
    converged = bool(info["converged"] and float(info["rel_residual_rot"]) <= TOL)
    write_json(args.out / "stage_relax.json", {
        "stage": "relax", "seed": 2, "checkpoint": CHECKPOINT,
        "checkpoint_sha256": sha(checkpoint),
        "converged_relative_equilibrium": converged,
        "relax_time": info["relax_time"], "relax_steps": info["relax_steps"],
        "rel_residual": float(info["rel_residual"]),
        "rel_residual_rot": float(info["rel_residual_rot"]), "omega": info["omega"],
        "newton_attempts": info["newton"], "cost": cost,
        "note": ("rel_residual_rot is the rotation-corrected residual; it is NOT the full steady "
                 "residual. Full acceptance requires the finish stage MAT read-back.")})
    print(json.dumps({"stage": "relax", "converged_relative_equilibrium": converged,
                      "relax_time": info["relax_time"], "rel_residual_rot": float(info["rel_residual_rot"]),
                      "cost": cost}), flush=True)
    return 0 if converged else 3


def finish_stage(args):
    """Checkpoint -> orient_polish -> newton_polish -> certified float64 MAT."""
    import numpy as np
    from turing_rd import STRIPES
    from turing_rd.diagnostics import pattern_stats
    from turing_rd.result_io import read_mat_fields, save_certified_mat, verify_mat
    from turing_rd.steady import newton_polish, orient_polish, rotation_generators, rotation_split

    started, cpu = time.monotonic(), time.process_time()
    checkpoint = args.out / CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing relax-stage checkpoint {checkpoint}")
    prov, stage = load_json(args.out / "provenance.json"), load_json(args.out / "stage_relax.json")
    if stage.get("checkpoint_sha256") != sha(checkpoint):
        raise ValueError("Checkpoint SHA256 does not match stage_relax.json")
    if not stage.get("converged_relative_equilibrium"):
        raise ValueError("Relax stage did not converge to a relative equilibrium; finish is not attempted")
    geom = geometry()
    with np.load(checkpoint) as saved:
        u = np.array(saved["u"], dtype=np.float64)
        v = np.array(saved["v"], dtype=np.float64)
        omega = np.array(saved["omega"], dtype=np.float64)
    if u.shape != (geom.n,) or v.shape != (geom.n,) or omega.shape != (3,):
        raise ValueError("Checkpoint fields do not match sphere6 float64 geometry")
    T = rotation_generators(geom)
    print("Following the relative-equilibrium family to zero drift (orient_polish)", flush=True)
    uo, vo, omo, ho = orient_polish(geom, STRIPES, u, v, omega, tol=TOL, T=T, verbose=True)
    un, vn, hn = newton_polish(geom, STRIPES, uo, vo, tol=TOL)
    if hn[-1]["rel_residual"] < TOL:              # a true steady state
        uo, vo = un, vn
    rs = rotation_split(geom, STRIPES, uo, vo, T)
    u, v = uo, vo
    orient = {"steps": len(ho) - 1, "rotated_rad": float(sum(h["theta"] for h in ho)),
              "from": float(ho[0]["rel_residual"]),
              "to": float(min(h["rel_residual"] for h in ho)),
              "newton_to": float(hn[-1]["rel_residual"]),
              "wall_s": float(ho[-1]["wall_s"] + hn[-1]["wall_s"]), "history": ho}
    info = {"relax_time": stage["relax_time"], "relax_steps": stage["relax_steps"],
            "rel_residual_relax": float(stage["rel_residual"]),
            "rel_residual_rot_relax": float(stage["rel_residual_rot"]),
            "rel_residual": rs["rel_residual"], "rel_residual_rot": rs["rel_residual_rot"],
            "omega": rs["omega"], "steady": bool(rs["rel_residual"] < TOL),
            "converged": bool(stage["converged_relative_equilibrium"]),
            "orient": orient, "newton": [], "newton_after_orient": hn}
    cost = {"orient_wall_seconds": orient["wall_s"],
            "finish_wall_seconds": time.monotonic() - started,
            "finish_cpu_seconds": time.process_time() - cpu,
            "relax_wall_seconds": stage["cost"]["relax_wall_seconds"]}
    write_json(args.out / "solver_info.json", info)
    prov["solver_cost"] = cost
    prov["actual_relax_time"] = stage["relax_time"]
    prov["actual_relax_steps"] = stage["relax_steps"]
    prov["rel_residual_rot_relax"] = stage["rel_residual_rot"]
    certificate = save_certified_mat(args.out / "solution.mat", geom, STRIPES, u, v,
                                     seed=2, tol=TOL, provenance=prov)
    # Re-read the actual MAT and independently repeat verification; no in-memory-only acceptance.
    ur, vr = read_mat_fields(args.out / "solution.mat", geom.n)
    if not np.array_equal(ur, u) or not np.array_equal(vr, v):
        raise RuntimeError("Saved MAT values differ from the solver's float64 fields")
    repeated = verify_mat(args.out / "solution.mat", geom, STRIPES, seed=2, tol=TOL)
    if repeated["mat_sha256"] != certificate["mat_sha256"]:
        raise RuntimeError("MAT changed during verification")
    np.savez_compressed(args.out / "final.npz", u=ur, v=vr,
                        mat_sha256=certificate["mat_sha256"], dtype="float64")
    shape = pattern_stats(geom, vr)
    accepted = bool(certificate["accepted"] and repeated["accepted"] and
                    shape.get("type") == "stripes_or_maze" and shape.get("n_components") == 7)
    full_cost = {"solve_wall_seconds": cost["relax_wall_seconds"] + cost["finish_wall_seconds"],
                 "solve_cpu_seconds": stage["cost"]["relax_cpu_seconds"] + cost["finish_cpu_seconds"],
                 **cost,
                 "total_worker_wall_seconds": time.monotonic() - started,
                 "note": "finish-stage process wall time; relax wall time comes from the relax stage"}
    summary = {"pipeline": "i02_historical_seed2_reproduction", "surface": "sphere6", "seed": 2,
               **STRIPES.as_dict(), "mat_certificate": certificate,
               "mat_readback_accepted": certificate["accepted"],
               "rel_residual": certificate["residual"]["value"],
               "pattern": shape, "candidate_generation_accepted": accepted,
               "solver_info": info, "cost": full_cost, "provenance": prov,
               "stages": {"relax": "stage_relax.json", "checkpoint": CHECKPOINT,
                          "checkpoint_sha256": stage["checkpoint_sha256"]},
               "full_discrete_stability": "not_yet_evaluated",
               "note": ("Generation acceptance is I01 full-residual/structure/precision plus "
                        "seven-band shape; no stability assertion.")}
    write_json(args.out / "summary.json", summary)
    print(json.dumps({"candidate_generation_accepted": accepted,
                      "residual": summary["rel_residual"],
                      "n_components": shape.get("n_components"),
                      "rel_residual_rot_relax": stage["rel_residual_rot"],
                      "cost": full_cost}), flush=True)
    return 0 if accepted else 2


def worker(args):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(args.out / "runtime_cache" / "matplotlib")
    sys.path.insert(0, str(ROOT))
    if args.stage == "finish":
        return finish_stage(args)
    code = relax_stage(args)
    if code or args.stage == "relax":
        return code
    return finish_stage(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed2", action="store_true", required=True, help="Run only historical candidate seed2")
    ap.add_argument("--stage", choices=("all", "relax", "finish"), default="all",
                    help="relax saves the float64 checkpoint before orient_polish; finish resumes from it")
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
    if args.stage == "finish":
        if not args.out.is_dir():
            ap.error("--stage finish requires the completed relax-stage directory")
        if not (args.out / CHECKPOINT).is_file():
            ap.error(f"--stage finish requires {CHECKPOINT} from the relax stage")
        names = {"log": "finish.log", "manifest": "finish_manifest.json",
                 "snapshot": "entrypoint_finish.py", "process": "worker_finish.json"}
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        names = {"log": "generation.log", "manifest": "run_manifest.json",
                 "snapshot": "entrypoint_used.py", "process": "worker_process.json"}
    start = time.monotonic()
    status = {"stage": args.stage, "seed": 2, "timeout_seconds": args.timeout,
              "success": False, "timed_out": False, "stability_status": "not_evaluated_by_generator"}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    # Preserve the exact launch source even when later engineering edits are made.
    with (args.out / names["snapshot"]).open("xb") as snapshot:
        snapshot.write(Path(__file__).read_bytes())
    proc = None
    try:
        with (args.out / names["log"]).open("x", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--seed2",
                                     "--stage", args.stage, "--worker", "--out", str(args.out),
                                     "--timeout", str(args.timeout)],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status["worker_launcher_pid"] = proc.pid
            write_json(args.out / names["process"], {
                "worker_launcher_pid": proc.pid, "parent_pid": os.getpid(), "stage": args.stage,
                "started_unix": time.time(), "timeout_seconds": args.timeout,
                "entrypoint_snapshot_sha256": sha(args.out / names["snapshot"]),
                "windows_timeout_policy": "taskkill of this Popen PID and its descendants only"})
            try:
                proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                status.update(timed_out=True,
                              failure_reason="Hard stage wall cap reached; no pass is certified.")
                status["timeout_cleanup"] = stop_worker_tree(proc)
        status.update(returncode=proc.returncode, success=proc.returncode == 0)
        if status["timed_out"]:
            status["success"] = False
        elif proc.returncode == 3:
            status["failure_reason"] = ("Relax stage did not reach a relative equilibrium; "
                                        "checkpoint is still saved for inspection.")
        elif proc.returncode:
            status["failure_reason"] = ("Worker failed or full-residual/float64/seven-band acceptance "
                                        "failed; inspect the stage log and any saved candidate.")
    except Exception as exc:
        status["failure_reason"] = repr(exc)
    finally:
        # Also clean up when PID logging or another parent operation fails after Popen.
        if proc is not None and proc.poll() is None:
            status["exception_cleanup"] = stop_worker_tree(proc)
    status["parent_wall_seconds"] = time.monotonic() - start
    status["command"] = [sys.executable, "scripts/i02_generate_candidate.py", "--seed2",
                         "--stage", args.stage, "--out", str(args.out), "--timeout", str(args.timeout)]
    write_json(args.out / names["manifest"], status)
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
