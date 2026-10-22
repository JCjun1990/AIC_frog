"""Recover I01 task artifacts with bounded Newton polishing, without training or time integration.

    python scripts/recover_float64_results.py --out results/i01_float64

Each task runs sequentially in a child process with a wall-clock timeout and an iteration cap.
The immutable input MAT/summary are hashed; all output goes to a new directory. A precision/
structure/full-residual certificate is not a linear-stability certificate (I02 is separate).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT_BASE = "065396591aef328c3484cb47e196cf5f8be71b6c"
TASKS = ("frog_spots", "frog_stripes", "sphere6_spots", "sphere6_stripes")
CODE_FILES = ("scripts/recover_float64_results.py", "turing_rd/result_io.py",
              "turing_rd/steady.py", "turing_rd/diagnostics.py", "turing_rd/geometry.py",
              "turing_rd/mesh.py", "turing_rd/solver.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def worker(args):
    # Set before importing NumPy/SciPy/Torch. No worker pool, GPU, rollout, or model loading.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLCONFIGDIR"] = str(args.out / "runtime_cache" / "matplotlib")
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import scipy.io as sio
    from turing_rd import Geometry, GSParams, icosphere, load_frog
    from turing_rd.diagnostics import pattern_stats
    from turing_rd.result_io import save_certified_mat, verify_mat
    from turing_rd.steady import newton_polish
    from turing_rd.viz import render_field

    task = args.worker
    source = args.source / task
    destination = args.out / "tasks" / task
    destination.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    original = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    surface, pattern = task.rsplit("_", 1)
    if original.get("surface") != surface:
        raise ValueError("Source summary surface does not match task")
    if not all(isinstance(original.get(key), (int, float)) for key in ("F", "k", "seed")):
        raise ValueError("Source summary must specify F, k and seed")
    params = GSParams(original["F"], original["k"])
    mesh = load_frog(ROOT / "frog.mat") if surface == "frog" else icosphere(6)
    geometry = Geometry(mesh, laplacian="robust", cache_dir=None)
    provenance = {"audit_base": AUDIT_BASE,
                  "entrypoint": "scripts/recover_float64_results.py",
                  "source_mat": str(source / "solution.mat"),
                  "source_mat_sha256": sha(source / "solution.mat"),
                  "source_summary_sha256": sha(source / "summary.json"),
                  "source_initial_image_sha256": sha(source / "ai_initial_v.png"),
                  "frog_input_sha256": sha(ROOT / "frog.mat"),
                  "code_sha256": {name: sha(ROOT / name) for name in CODE_FILES},
                  "method": "bounded_newton_from_quantized_mat",
                  "max_newton_iterations": args.max_iter,
                  "task_wall_limit_seconds": args.seconds_per_task,
                  "operator": "robust", "robust_mollify_factor": 1e-5,
                  "threads": 1, "time_integration_steps": 0,
                  "precision_recovery": "newton_correction; dtype conversion alone is not recovery"}
    versions = {}
    for name in ("numpy", "scipy", "torch", "robust-laplacian", "matplotlib"):
        versions[name] = importlib.metadata.version(name)
    provenance["environment"] = {"python": sys.version, "platform": sys.platform, "packages": versions}
    before = verify_mat(source / "solution.mat", geometry, params,
                        seed=original["seed"], tol=args.tol, provenance=provenance)
    data = sio.loadmat(source / "solution.mat", struct_as_record=False)["Solution"][0, 0]
    u, v = data.u[:, 0].copy(), data.v[:, 0].copy()
    # No original full-precision NPZ is present in the fixed repository. The input is explicitly
    # acknowledged as quantized; actual Newton corrections must produce the recovered state.
    before_pattern = pattern_stats(geometry, v)
    up, vp, history = newton_polish(geometry, params, u, v, tol=args.tol, max_iter=args.max_iter)
    provenance["newton_iterations"] = len(history) - 1
    provenance["max_abs_newton_correction_u"] = float(np.max(np.abs(up - u)))
    provenance["max_abs_newton_correction_v"] = float(np.max(np.abs(vp - v)))
    certificate = save_certified_mat(destination / "solution.mat", geometry, params, up, vp,
                                     seed=original["seed"], tol=args.tol, provenance=provenance)
    readback = sio.loadmat(destination / "solution.mat", struct_as_record=False)["Solution"][0, 0]
    up, vp = readback.u[:, 0], readback.v[:, 0]
    after_pattern = pattern_stats(geometry, vp)
    same_shape = (before_pattern.get("type") == after_pattern.get("type") and
                  before_pattern.get("n_components") == after_pattern.get("n_components"))
    summary = {"pipeline": "i01_bounded_newton_recovery", "surface": surface, "pattern": pattern,
               **params.as_dict(), "seed": original["seed"],
               "mat_certificate": certificate, "mat_readback_accepted": certificate["accepted"],
               "rel_residual": certificate["residual"]["value"],
               "residual_formula": certificate["residual"]["formula"],
               "before_mat_validation": before, "newton_history": history,
               "pattern_before": before_pattern, "pattern_after": after_pattern,
               "shape_preserved": same_shape,
               "stability_status": "not_recomputed_I01_only",
               "full_discrete_stability": "not_certified_by_I01",
               "source_stability_record": {key: original[key] for key in
                    ("stable", "max_re_lambda", "rotation_eigenvalues", "rotation_overlap") if key in original},
               "legacy_residual_before_quantization": original.get("rel_residual"),
               "provenance": provenance,
               "note": "I01 acceptance covers MAT structure, precision and full PDE residual only; no new stability claim."}
    np.savez_compressed(destination / "final.npz", u=up, v=vp, vertices=geometry.V, faces=geometry.F,
                        state_dtype="float64", source_mat_sha256=certificate["mat_sha256"])
    shutil.copyfile(source / "ai_initial_v.png", destination / "ai_initial_v.png")
    render_field(geometry, vp, destination / "final_v.png",
                 title=f"{task}: I01 MAT readback full residual={summary['rel_residual']:.3e}; stability not rechecked")
    summary["wall_seconds"] = time.monotonic() - start
    write_json(destination / "summary.json", summary)
    print(json.dumps({"task": task, "accepted": certificate["accepted"],
                      "before": before["residual"]["value"], "after": summary["rel_residual"],
                      "iterations": len(history)-1, "shape_preserved": same_shape,
                      "wall_seconds": summary["wall_seconds"]}), flush=True)
    return 0 if certificate["accepted"] and same_shape else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results" / "ai_generation" / "tasks")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "i01_float64")
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--seconds-per-task", type=float, default=120.0)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument("--worker", choices=TASKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source, args.out = args.source.resolve(), args.out.resolve()
    if not 1 <= args.max_iter <= 15 or not 0 < args.seconds_per_task <= 300 or not 0 < args.tol <= 1e-11:
        parser.error("Use 1..15 Newton iterations, 0..300 seconds per task and 0<tol<=1e-11")
    if args.out == args.source or args.out.is_relative_to(args.source) or args.source.is_relative_to(args.out):
        parser.error("Output and immutable source directories must be separate")
    if args.worker:
        return worker(args)
    if args.out.exists():
        parser.error("Output already exists; select a new directory (nothing is overwritten)")
    for task in TASKS:
        for filename in ("solution.mat", "summary.json", "ai_initial_v.png"):
            if not (args.source / task / filename).is_file():
                parser.error(f"Missing input: {args.source / task / filename}")
    args.out.mkdir(parents=True, exist_ok=False)
    initial_hashes = {str(p.relative_to(args.source)): sha(p) for task in TASKS
                      for p in sorted((args.source / task).rglob("*")) if p.is_file()}
    outcomes = []
    for task in TASKS:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", task,
                   "--source", str(args.source), "--out", str(args.out), "--max-iter", str(args.max_iter),
                   "--seconds-per-task", str(args.seconds_per_task), "--tol", str(args.tol)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        start = time.monotonic()
        result = {"task": task, "command": command}
        try:
            process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                     encoding="utf-8", timeout=args.seconds_per_task)
            result.update(returncode=process.returncode, accepted=process.returncode == 0)
            (args.out / f"{task}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
            if process.returncode:
                result["failure_reason"] = f"worker_exit_code_{process.returncode}; see {task}.log"
                summary_path = args.out / "tasks" / task / "summary.json"
                if summary_path.exists():
                    failed = json.loads(summary_path.read_text(encoding="utf-8"))
                    reasons = list(failed.get("mat_certificate", {}).get("failure_reasons", []))
                    if not failed.get("shape_preserved", False):
                        reasons.append("shape_changed")
                    result["failure_reasons"] = reasons
        except subprocess.TimeoutExpired as error:
            result.update(returncode=None, accepted=False, failure_reason="wall_clock_budget_exhausted")
            parts = (error.stdout or b"", error.stderr or b"")
            output = b"".join(p.encode("utf-8") if isinstance(p, str) else p for p in parts)
            (args.out / f"{task}.log").write_bytes(output)
        result["wall_seconds"] = time.monotonic() - start
        outcomes.append(result)
        print(json.dumps(result), flush=True)
    source_changes = []
    for rel, digest in initial_hashes.items():
        try:
            if sha(args.source / rel) != digest:
                source_changes.append({"path": rel, "reason": "sha256_changed"})
        except OSError as error:
            source_changes.append({"path": rel, "reason": f"{type(error).__name__}: {error}"})
    source_unchanged = not source_changes
    write_json(args.out / "run_manifest.json", {"audit_base": AUDIT_BASE,
               "entrypoint": "scripts/recover_float64_results.py", "source": str(args.source),
               "source_hashes": initial_hashes, "source_unchanged": source_unchanged,
               "source_changes": source_changes,
               "max_iterations_per_task": args.max_iter, "seconds_per_task": args.seconds_per_task,
               "sequential_workers": 1, "outcomes": outcomes,
               "accepted_I01": source_unchanged and all(r["accepted"] for r in outcomes),
               "stability_status": "not_evaluated_I02_not_started"})
    return 0 if source_unchanged and all(r["accepted"] for r in outcomes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
