"""Baseline generation (tasks 1 and 2): Gray-Scott on the unit sphere or the frog.

Protocol (defaults):
    initial state  organisers' suggested IC, z scaled by the surface half-height (no-op on the unit
                   sphere), with localised (1/2, 1/4) seeds and 5 % smooth noise, fixed seed
    phase 1        SBDF2, dt = 1 up to t = 20 000 (resolves the transient)
    phase 2        SBDF2, dt = 4 up to t = 400 000 or until rms du/dt, dv/dt < tol (slow relaxation;
                   the fixed points of the scheme are exact steady states for any dt)
    polish         pseudo-transient continuation / Newton to a round-off steady state, attempted
                   only when the time integration has converged (near the Newton basin)
    stability      eigenvalues of the Jacobian nearest 0 (shift-invert ARPACK)

Writes to results/baseline/<surface>_<pattern>/:
    solution.mat   Solution.u, Solution.v (same layout as the organisers' Solution.mat)
    final.npz      u, v, vertices, faces;  log.csv convergence record;  summary.json everything else
    field_v.png    rendered v;  convergence.png residual history;  field_v.ply coloured mesh

Example:
    python scripts/run_baseline.py --surface frog --pattern spots
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turing_rd import Geometry, GrayScottSolver, GSParams, SPOTS, STRIPES, icosphere, load_frog  # noqa: E402
from turing_rd.diagnostics import linear_stability, pattern_stats, pde_residual, scales  # noqa: E402
from turing_rd.initial import perturbed_ic, suggested_ic  # noqa: E402
from turing_rd.mesh import save_ply  # noqa: E402
from turing_rd.steady import ptc_polish  # noqa: E402
from turing_rd.viz import plot_history, render_field  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", choices=["sphere", "frog"], required=True)
    ap.add_argument("--pattern", choices=["spots", "stripes"], required=True)
    ap.add_argument("--level", type=int, default=6, help="icosphere subdivision level (sphere only)")
    ap.add_argument("--frog", default=str(ROOT / "frog.mat"))
    ap.add_argument("--laplacian", choices=["robust", "cotan"], default="robust")
    ap.add_argument("--ic", choices=["seeded", "additive", "pearson", "suggested", "target"], default="seeded",
                    help="'target': start from Solution.mat made admissible (h-scale smoothing, u in [0, 1], v >= 0)")
    ap.add_argument("--no-normalize-z", action="store_true", help="use raw z in the suggested IC")
    ap.add_argument("--eps", type=float, default=0.05, help="perturbation amplitude")
    ap.add_argument("--noise-length", type=float, default=0.05, help="perturbation correlation length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scheme", choices=["sbdf1", "sbdf2"], default="sbdf2")
    ap.add_argument("--dt", type=float, default=1.0, help="time step of phase 1")
    ap.add_argument("--t-switch", type=float, default=20000, help="end of phase 1")
    ap.add_argument("--dt-late", type=float, default=4.0, help="time step of phase 2")
    ap.add_argument("--t-max", type=float, default=400000)
    ap.add_argument("--t-min", type=float, default=2000)
    ap.add_argument("--tol", type=float, default=1e-7, help="stop when rms du/dt and dv/dt fall below this")
    ap.add_argument("--polish-tol", type=float, default=1e-11, help="target relative steady residual")
    ap.add_argument("--polish-from", type=float, default=1e-4,
                    help="also try the polish when the time integration stops above tol but below this residual")
    ap.add_argument("--log-every", type=float, default=500, help="log interval in time units")
    ap.add_argument("--backend", choices=["lu", "cg"], default="lu")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-stability", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline"))
    ap.add_argument("--F", type=float, default=None, help="override F of the chosen pattern")
    ap.add_argument("--k", type=float, default=None, help="override k")
    ap.add_argument("--Du", type=float, default=None, help="override Du")
    ap.add_argument("--Dv", type=float, default=None, help="override Dv")
    ap.add_argument("--name", default=None, help="output sub-directory (default <surface>_<pattern>)")
    args = ap.parse_args(argv)

    params = SPOTS if args.pattern == "spots" else STRIPES
    params = GSParams(*(o if o is not None else d for o, d in
                        zip((args.F, args.k, args.Du, args.Dv), (params.F, params.k, params.Du, params.Dv))))
    mesh = icosphere(args.level) if args.surface == "sphere" else load_frog(args.frog)
    geom = Geometry(mesh, laplacian=args.laplacian, cache_dir=ROOT / "results" / "cache")
    out = Path(args.out) / (args.name or f"{args.surface}_{args.pattern}")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.surface}/{args.pattern}"

    nz = not args.no_normalize_z
    if args.ic == "target":
        if args.surface != "frog":
            raise SystemExit("--ic target needs --surface frog")
        S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
        A = (sp.diags(geom.M) - geom.h_med ** 2 * geom.L).tocsc()
        u0 = np.clip(spla.spsolve(A, geom.M * S["u"][0, 0].ravel()), 0, 1)
        v0 = np.maximum(spla.spsolve(A, geom.M * S["v"][0, 0].ravel()), 0)
    elif args.ic == "suggested":
        u0, v0 = suggested_ic(geom, normalize_z=nz)
    else:
        u0, v0 = perturbed_ic(geom, eps=args.eps, seed=args.seed, length=args.noise_length,
                              mode=args.ic, normalize_z=nz)

    dtype = torch.float64 if args.backend == "lu" else torch.float32
    rows, t0 = [], time.time()

    def make_logger(t_offset, step_offset, dt):
        def log(step, t, u, v, up, vp):
            r = pde_residual(geom, params, u, v, up, vp, dt if up is not None else None)
            s = pattern_stats(geom, v)
            row = {"phase": f"dt={dt:g}", "step": step_offset + step, "t": t_offset + t,
                   "wall_s": time.time() - t0, **r,
                   "type": s["type"], "n_components": s.get("n_components", 0),
                   "area_fraction": s.get("area_fraction", 0.0), "wavelength": s["wavelength"],
                   "compactness": s.get("compactness", 0.0), "v_min": s["v_min"], "v_max": s["v_max"]}
            rows.append(row)
            print(f"[{tag}] t={row['t']:8.0f} du/dt={row.get('dudt_rms', float('nan')):.2e} "
                  f"steady_rel={r['steady_rel']:.2e} {s['type']:15s} n={row['n_components']:3d} "
                  f"lam={s['wavelength']:.3f} ({row['wall_s']:.0f}s)", flush=True)
            return (row["t"] >= args.t_min and row.get("dudt_rms", 1) < args.tol
                    and row.get("dvdt_rms", 1) < args.tol)
        return log

    def converged_now():
        last = rows[-1] if rows else {}
        return bool(last.get("dudt_rms", 1) < args.tol and last.get("dvdt_rms", 1) < args.tol)

    # ---- phase 1 and phase 2 time integration ------------------------------------------------
    phases = [(args.dt, min(args.t_switch, args.t_max)), (args.dt_late, args.t_max)]
    u, v, t_now, steps_total = u0, v0, 0.0, 0
    with torch.no_grad():
        for dt, t_end in phases:
            if t_now >= t_end or (rows and converged_now()):
                continue
            solver = GrayScottSolver(geom, dt=dt, scheme=args.scheme, backend=args.backend,
                                     device=args.device, dtype=dtype)
            n = int(round((t_end - t_now) / dt))
            every = max(1, int(round(args.log_every / dt)))
            n_before = len(rows)
            u, v = solver.rollout(u, v, params, n, record_every=every,
                                  callback=make_logger(t_now, steps_total, dt))
            last_step = rows[-1]["step"] - steps_total if len(rows) > n_before else n
            steps_total += last_step
            t_now += last_step * dt
            u, v = u.double().cpu().numpy(), v.double().cpu().numpy()
    wall_time_integration = time.time() - t0
    converged = converged_now()

    # ---- steady-state polish -----------------------------------------------------------------
    polish = None
    near = rows and rows[-1]["steady_rel"] < args.polish_from
    if converged or near:
        tp = time.time()
        up, vp, hist = ptc_polish(geom, params, u, v, tol=args.polish_tol, dtau0=1e3, max_iter=30)
        ok = hist[-1]["rel_residual"] < 0.1 * hist[0]["rel_residual"]
        same = pattern_stats(geom, vp).get("n_components") == pattern_stats(geom, v).get("n_components")
        polish = {"history": hist, "accepted": bool(ok and same), "seconds": time.time() - tp,
                  "max_abs_change_v": float(np.abs(vp - v).max())}
        if polish["accepted"]:
            u, v = up, vp
        print(f"[{tag}] polish: rel residual {hist[0]['rel_residual']:.2e} -> {hist[-1]['rel_residual']:.2e} "
              f"in {len(hist) - 1} iterations, accepted={polish['accepted']}", flush=True)

    final_stats = pattern_stats(geom, v)
    final_res = pde_residual(geom, params, u, v)
    stab = None
    if not args.no_stability:
        ts = time.time()
        stab = linear_stability(geom, params, u, v)
        stab["seconds"] = time.time() - ts
    last = rows[-1] if rows else {}
    summary = {
        "surface": args.surface, "pattern": args.pattern, "params": params.as_dict(),
        "mesh": geom.summary(), "args": vars(args), "scales": scales(geom, params, args.dt),
        "time": {"t_final": t_now, "steps": steps_total, "wall_seconds_time_integration": wall_time_integration,
                 "wall_seconds_total": time.time() - t0, "device": args.device, "backend": args.backend,
                 "torch": torch.__version__},
        "converged_time_integration": converged,
        "final_rates": {k: last.get(k) for k in ("dudt_rms", "dvdt_rms")},
        "polish": polish, "final_residual": final_res, "final_pattern": final_stats, "linear_stability": stab,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    with open(out / "log.csv", "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez(out / "final.npz", u=u, v=v, vertices=geom.V, faces=geom.F, u0=u0, v0=v0)
    sio.savemat(out / "solution.mat", {"Solution": {"u": u[:, None], "v": v[:, None]}})
    title = (f"{args.surface} · {args.pattern} (F={params.F}, k={params.k}) · t={t_now:.0f} · "
             f"{final_stats['type']}, n={final_stats.get('n_components', 0)}, λ={final_stats['wavelength']:.3f}")
    views = ("top", "iso", "iso_back") if args.surface == "frog" else ("top", "front", "iso")
    render_field(geom, v, out / "field_v.png", title=title, views=views, vmin=0.0, vmax=max(0.3, v.max()))
    render_field(geom, v0, out / "initial_v.png", title=f"{tag} initial v", views=views, vmin=0.0, vmax=0.5)
    plot_history(rows, out / "convergence.png", title=f"{tag} convergence",
                 polish=polish["history"] if polish and polish["accepted"] else None)
    save_ply(out / "field_v.ply", geom.mesh, v, vmin=0.0, vmax=max(0.3, v.max()))
    print(json.dumps({"out": str(out), "converged": converged, "t_final": t_now, "type": final_stats["type"],
                      "n": final_stats.get("n_components"), "wavelength": final_stats["wavelength"],
                      "steady_rel": final_res["steady_rel"],
                      "max_real_eig": stab["max_real"] if stab else None,
                      "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
