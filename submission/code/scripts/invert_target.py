"""One-command inversion of Solution.mat for F and k (Du, Dv fixed at the given spot values).

Scoring (organisers' specialist teacher, 2026-10-20): pattern type correct, feature match
(characteristic wavelength, orientation, defects), PDE-residual convergence; no pixel alignment.

Stages (fixed seeds throughout; outputs in results/inversion/):
    1 preprocess   heat-kernel low-pass with the cut-off taken from the residual curve, then clipping
                   to u in [0, 1], v >= 0 (turing_rd/preprocess.py)
    2 closed form  F0, k0 from the integral identities on the preprocessed target
    3 type map     coarse grid over the classical Gray-Scott range (Pearson 1993), one initial state,
                   t = 5000: where does the model make spots at all?
    4 grid         fine (F, k) grid around (F0, k0); at every point the same S random initial states
                   (common random numbers) are integrated to steady states and polished by PTC/Newton;
                   J = median over the seeds of the feature distance to the target (type, wavelength,
                   orientation, defects; turing_rd/inverse.py)
    5 surrogate    Gaussian-process regression of J over (F, k): the minimum of the posterior mean is
                   the estimate; the uncertainty band is where the mean stays within 2 bootstrap
                   standard errors of the minimum; a bootstrap over seeds gives the spread of the
                   minimiser; the cut-off sensitivity is J recomputed at 0.8x and 1.25x the cut-off
    6 final        steady states at the estimate and, for reference, at the given spot parameters,
                   polished to 1e-12, with linear stability and the feature table

    python scripts/invert_target.py                     full run (about 15 min, Mac CPU, 8 workers)
    python scripts/invert_target.py --quick             plumbing check (3 x 3 grid, 2 seeds)
"""
from __future__ import annotations

import os

for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import SPOTS, Geometry, GSParams, load_frog  # noqa: E402
from turing_rd.inverse import OBJECTIVE, TYPE_PENALTY, feature_distance, is_steady, steady_states  # noqa: E402
from turing_rd.preprocess import HeatLowPass, admissible, integral_identities, select_cutoff  # noqa: E402

OUT = ROOT / "results" / "inversion"
GIVEN = (SPOTS.F, SPOTS.k)
SCALAR_SKIP = ("spot_Q", "spot_area")
W: dict = {}                                          # per-worker state


# ---------------------------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------------------------
def _init_worker(kappas):
    from turing_rd.features import FeatureExtractor
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    W.update(g=g, fx=FeatureExtractor(g, extended=True), lp={kh: HeatLowPass(g, kh) for kh in kappas})


def _scalars(d):
    return {k: (v if isinstance(v, str) else float(v)) for k, v in d.items() if k not in SCALAR_SKIP}


def features_of(u, v, kh):
    """Measurement operator (low-pass at the target's cut-off, projection on the physical range) + features."""
    return _scalars(W["fx"].describe(*admissible(W["lp"][kh](u), W["lp"][kh](v))))


def job_types(args):
    """Coarse type map: one column per (F, k), no polish."""
    Fs, ks, seed, t_end = args
    g = W["g"]
    res = steady_states(g, Fs, ks, np.full(len(Fs), seed), SPOTS.Du, SPOTS.Dv, t_max=t_end, polish=False,
                        chunk=t_end)
    out = []
    for r in res:
        d = W["fx"].describe(r["u"], r["v"])
        out.append({"F": r["F"], "k": r["k"], "type": d["type_detail"], "n_spots": float(d.get("n_spots", 0) or 0),
                    "wavelength": float(d.get("wavelength", np.nan)), "v_max": float(r["v"].max()),
                    "rate": r["rate"], "status": r["status"]})
    return out


def job_grid(args):
    """All seeds at one (F, k): steady states, features at every cut-off; states returned as float32."""
    F, k, seeds, cfg = args
    res = steady_states(W["g"], F, k, seeds, SPOTS.Du, SPOTS.Dv, t_max=cfg["t_max"], dt=cfg["dt"],
                        rate_tol=cfg["rate_tol"], polish_tol=cfg["polish_tol"])
    out = []
    for r in res:
        feats = {str(kh): features_of(r["u"], r["v"], kh) for kh in W["lp"]}
        out.append({**{key: r[key] for key in ("F", "k", "seed", "t", "rate", "polished", "residual", "ptc_iters",
                                               "status", "wall_s")},
                    "features": feats, "u": r["u"].astype(np.float32), "v": r["v"].astype(np.float32)})
    return out


def job_final(args):
    """One seed at one (F, k): steady state polished to round-off, residuals, linear stability, features."""
    from turing_rd.diagnostics import linear_stability, pde_residual
    from turing_rd.steady import ptc_polish
    label, F, k, seed, cfg, kh = args
    g = W["g"]
    p = GSParams(F, k, SPOTS.Du, SPOTS.Dv)
    r = steady_states(g, F, k, [seed], p.Du, p.Dv, t_max=cfg["t_max_final"], dt=cfg["dt"],
                      rate_tol=cfg["rate_tol"], polish_tol=cfg["polish_tol"])[0]
    u, v, hist = r["u"], r["v"], []
    if r["status"] != "uniform":
        u, v, hist = ptc_polish(g, p, u, v, tol=1e-12, dtau0=1e3, max_iter=40)
    rel = hist[-1]["rel_residual"] if hist else r["residual"]
    stab = linear_stability(g, p, u, v, k=8) if r["status"] != "uniform" and rel < 1e-8 else None
    return {"label": label, "F": F, "k": k, "seed": seed, "t": r["t"], "status": r["status"],
            "polished_in_loop": r["polished"], "rel_residual": rel, "ptc_extra_iters": max(len(hist) - 1, 0),
            "steady_rel": pde_residual(g, p, u, v)["steady_rel"], "stability": stab,
            "features": features_of(u, v, kh), "features_raw": _scalars(W["fx"].describe(u, v)),
            "u": u, "v": v, "wall_s": r["wall_s"]}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def boot_se(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return float(np.std([np.median(rng.choice(x, len(x))) for _ in range(n)]))


def fit_gp(X, y, se=None):
    """GP regression (RBF); with `se` the per-point noise is the known standard error, else it is fitted."""
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    kern = ConstantKernel(1.0, (1e-2, 1e2)) * RBF([1.5, 1.5], (0.5, 30.0))
    if se is None:
        return GaussianProcessRegressor(kern + WhiteKernel(0.01, (1e-6, 1.0)), normalize_y=True,
                                        n_restarts_optimizer=4, random_state=0).fit(X, y)
    alpha = (np.maximum(se, 0.05) / max(np.std(y), 1e-12)) ** 2         # alpha acts on the normalised targets
    return GaussianProcessRegressor(kern, alpha=alpha, normalize_y=True, n_restarts_optimizer=4,
                                    random_state=0).fit(X, y)


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=6, help="random initial states per candidate (CRN)")
    ap.add_argument("--grid", type=int, nargs=2, default=[7, 7], help="points in F and k")
    ap.add_argument("--dF", type=float, default=0.002, help="grid step in F")
    ap.add_argument("--dk", type=float, default=0.001, help="grid step in k")
    ap.add_argument("--t-max", type=float, default=40000.0, help="integration cap per candidate")
    ap.add_argument("--t-max-final", type=float, default=80000.0)
    ap.add_argument("--dt", type=float, default=8.0, help="SBDF2 step after the t <= 500 transient (dt = 1)")
    ap.add_argument("--rate-tol", type=float, default=1e-6)
    ap.add_argument("--polish-tol", type=float, default=1e-10)
    ap.add_argument("--kappa-half", type=float, default=None, help="override the chosen low-pass cut-off")
    ap.add_argument("--no-typemap", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse grid results of an identical earlier run")
    ap.add_argument("--quick", action="store_true", help="tiny run to check the plumbing")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.grid, args.t_max, args.t_max_final = 2, [3, 3], 6000.0, 6000.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "run.log", "w")
    T0 = time.time()
    walls = {}

    # ---- 1-2: preprocessing and closed form -------------------------------------------------------
    t = time.time()
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ut, vt = S["u"][0, 0].ravel().astype(float), S["v"][0, 0].ravel().astype(float)
    kh_auto, curve = select_cutoff(g, ut, vt, SPOTS.Du, SPOTS.Dv)
    kh = args.kappa_half or kh_auto
    kappas = [kh, round(0.8 * kh, 3), round(1.25 * kh, 3)]
    lp = HeatLowPass(g, kh)
    up, vp = admissible(lp(ut), lp(vt))
    F0, k0 = integral_identities(g, up, vp)
    raw_cf = integral_identities(g, *admissible(ut, vt))
    cf_band = [r for r in curve if 0.8 * kh - 1e-9 <= r["kappa_half"] <= 1.25 * kh + 1e-9]
    with open(out / "cutoff_curve.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(curve[0]))
        wr.writeheader()
        wr.writerows(curve)
    walls["preprocess"] = time.time() - t
    log(f"[1] cut-off kappa_half = {kh:g} (auto {kh_auto:g}, Gaussian sigma {lp.sigma:.3f}); "
        f"v<0 area {g.M[vt < 0].sum() / g.area:.3f} -> 0; v max {vt.max():.3f} -> {vp.max():.3f}", fh)
    log(f"[2] closed form F0 = {F0:.5f}, k0 = {k0:.5f} (raw, only clipped: {raw_cf[0]:.5f}, {raw_cf[1]:.5f}); "
        f"cut-off band 0.8-1.25x: F {min(r['F'] for r in cf_band):.5f}-{max(r['F'] for r in cf_band):.5f}, "
        f"k {min(r['k'] for r in cf_band):.5f}-{max(r['k'] for r in cf_band):.5f}", fh)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, _init_worker, (kappas,))
    _init_worker(kappas)
    tgt = {str(kk): features_of(ut, vt, kk) for kk in kappas}
    tgt_raw = _scalars(W["fx"].describe(ut, vt))
    log(f"    target (preprocessed): type {tgt[str(kh)]['type']}, n {tgt[str(kh)]['n_spots']:.0f}, "
        f"wavelength {tgt[str(kh)]['wavelength']:.3f}, coherence {tgt[str(kh)]['coherence']:.3f}, "
        f"spot defects {tgt[str(kh)]['spot_defect_frac']:.2f}", fh)

    # ---- 3: coarse type map -----------------------------------------------------------------------
    types = []
    if not args.no_typemap:
        t = time.time()
        Fc = np.round(np.arange(0.010, 0.0701, 0.005 if not args.quick else 0.02), 4)
        kc = np.round(np.arange(0.040, 0.0701, 0.0025 if not args.quick else 0.01), 4)
        jobs = [(np.full(len(kc), F), kc, 1000, 5000.0) for F in Fc]
        for rows in pool.imap(job_types, jobs):
            types += rows
        with open(out / "type_map.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(types[0]))
            wr.writeheader()
            wr.writerows(types)
        walls["type_map"] = time.time() - t
        cnt = {}
        for r in types:
            cnt[r["type"]] = cnt.get(r["type"], 0) + 1
        log(f"[3] type map {len(Fc)} x {len(kc)}: {cnt} ({walls['type_map']:.0f}s)", fh)

    # ---- 4: fine grid with CRN seeds --------------------------------------------------------------
    t = time.time()
    nF, nk = args.grid
    Fg = F0 + args.dF * (np.arange(nF) - (nF - 1) / 2)
    kg = k0 + args.dk * (np.arange(nk) - (nk - 1) / 2)
    seeds = [1000 * (i + 1) for i in range(args.seeds)]
    cfg = {"t_max": args.t_max, "t_max_final": args.t_max_final, "dt": args.dt, "rate_tol": args.rate_tol,
           "polish_tol": args.polish_tol}
    grid_cfg = {k_: v_ for k_, v_ in cfg.items() if k_ != "t_max_final"}
    key = hashlib.sha1(json.dumps([list(Fg), list(kg), seeds, grid_cfg, kappas], sort_keys=True).encode()).hexdigest()[:12]
    cache = out / "grid_cache.npz"
    grid = None
    if args.reuse and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["key"]) == key:
            grid = list(z["grid"])
            grid_wall = float(z["grid_wall"]) if "grid_wall" in z.files else float("nan")
            log(f"[4] reusing {len(grid)} grid evaluations from {cache.name} (computed in {grid_wall:.0f} s)", fh)
    if grid is None:
        jobs = [(float(F), float(k), seeds, cfg) for F in Fg for k in kg]
        grid = []
        for i, rows in enumerate(pool.imap_unordered(job_grid, jobs)):
            grid += rows
            log(f"    grid {i + 1}/{len(jobs)}: F={rows[0]['F']:.4f} k={rows[0]['k']:.4f} "
                f"polished {sum(r['polished'] for r in rows)}/{len(rows)}, t_end {max(r['t'] for r in rows):.0f}, "
                f"types {[r['features'][str(kh)]['type'] for r in rows]} ({time.time() - t:.0f}s)", fh)
        grid_wall = time.time() - t
        np.savez_compressed(cache, key=key, grid=np.array(grid, dtype=object), grid_wall=grid_wall)
    walls["grid"] = grid_wall                             # the original computation time when reused

    def J_table(kk):
        """Per (F, k) point: (seed, distance, per-criterion) for every seed; unsteady seeds get the penalty."""
        tab = {}
        for r in grid:
            if is_steady(r):
                d, per = feature_distance(r["features"][str(kk)], tgt[str(kk)])
            else:
                d, per = TYPE_PENALTY, {g_: TYPE_PENALTY for g_ in OBJECTIVE}
            tab.setdefault((round(r["F"], 7), round(r["k"], 7)), []).append((r["seed"], d, per))
        return tab

    def matrix(tab_):
        return np.array([[d for _, d, _ in sorted(tab_[p_])] for p_ in sorted(tab_)])   # (points, seeds)

    tab = J_table(kh)
    pts = sorted(tab)
    Dmat = matrix(tab)
    J = np.median(Dmat, 1)
    SE = np.array([boot_se(row) for row in Dmat])
    U = (Dmat < TYPE_PENALTY).mean(1)                        # share of seeds with a steady spot pattern
    X = np.array([((F - F0) / args.dF, (k - k0) / args.dk) for F, k in pts])
    polished = float(np.mean([r["polished"] for r in grid]))
    steady = float(np.mean([is_steady(r) for r in grid]))
    status = {}
    for r in grid:
        status[r["status"]] = status.get(r["status"], 0) + 1
    per_med = {g_: np.array([np.median([pc[g_] for _, _, pc in tab[p_]]) for p_ in pts]) for g_ in OBJECTIVE}
    with open(out / "grid_points.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["F", "k", "J", "J_se", "usable", "polished"] + [f"J_{g_}" for g_ in OBJECTIVE])
        for i, (F, k) in enumerate(pts):
            npol = sum(r["polished"] for r in grid if (round(r["F"], 7), round(r["k"], 7)) == (F, k))
            wr.writerow([F, k, J[i], SE[i], U[i], npol] + [per_med[g_][i] for g_ in OBJECTIVE])
    log(f"[4] grid {nF} x {nk} x {len(seeds)} seeds: polished {polished:.0%}, steady {steady:.0%}, status {status} "
        f"({walls['grid']:.0f}s)", fh)
    for F in sorted({p_[0] for p_ in pts}, reverse=True):
        log("    F=%.4f  " % F + "  ".join(f"{J[i]:4.2f}/{U[i]:.1f}" for i, p_ in enumerate(pts) if p_[0] == F) +
            "   (J / share of steady spot seeds; k = " + " ".join(f"{k:.4f}" for k in sorted({p_[1] for p_ in pts})) + ")", fh)

    # ---- 5: GP surrogate on the feasible region, PDE criterion, band, bootstrap, sensitivity ----
    # J_feat: median over the seeds with a steady spot pattern (the share of such seeds, U, decides
    # feasibility); J_pde: the target's integral-identity residual at (F, k) in units of its
    # preprocessing uncertainty (half the spread over the 0.8-1.25x cut-off band); J_total combines the
    # three feature criteria and the PDE criterion with equal weight.
    t = time.time()
    cf = {str(kk): next(r for r in curve if abs(r["kappa_half"] - kk) < 1e-9) for kk in kappas}
    sig_F = 0.5 * (max(r["F"] for r in cf_band) - min(r["F"] for r in cf_band)) / F0
    sig_S = 0.5 * (max(r["F"] + r["k"] for r in cf_band) - min(r["F"] + r["k"] for r in cf_band)) / (F0 + k0)
    n_crit = len(OBJECTIVE)

    def J_pde(F, k, kk=kh):
        c = cf[str(kk)]
        zF = (np.asarray(F) / c["F"] - 1) / sig_F
        zS = ((np.asarray(F) + np.asarray(k)) / (c["F"] + c["k"]) - 1) / sig_S
        return np.sqrt(0.5 * (zF ** 2 + zS ** 2))

    def J_total(jf, jp):
        return np.sqrt((n_crit * jf ** 2 + jp ** 2) / (n_crit + 1))

    feas = U >= 0.5
    fallback = feas.sum() < 3            # e.g. --quick: nothing steady yet; keep the plumbing running on raw medians
    if fallback:
        feas = np.ones_like(feas)
        log("    fewer than 3 feasible grid points: the surrogate falls back to medians including penalties "
            "(plumbing check only, not a result)", fh)

    def cond_median(D):
        if fallback:
            return np.median(D, 1)
        return np.array([np.median(row[row < TYPE_PENALTY]) if (row < TYPE_PENALTY).any() else np.nan for row in D])

    Jc = cond_median(Dmat)
    SEc = np.array([boot_se(row[row < TYPE_PENALTY]) if (row < TYPE_PENALTY).sum() > 1 else np.nan for row in Dmat])
    SEf = np.where(np.isfinite(SEc), SEc, np.nanmedian(SEc[feas]) if np.isfinite(SEc[feas]).any() else 0.1)
    gp = fit_gp(X[feas], Jc[feas], SEf[feas])
    gpu = fit_gp(X, U)
    nf = 121
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), nf)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), nf)
    XX = np.array([(a_, b_) for a_ in xs for b_ in ys])
    FF, KK = F0 + args.dF * XX[:, 0], k0 + args.dk * XX[:, 1]
    mask = np.ones(len(XX), bool) if fallback else gpu.predict(XX) >= 0.5
    jp = J_pde(FF, KK)

    def surfaces(model, kk=kh):
        mf = np.where(mask, model.predict(XX), np.nan)
        return mf, J_total(mf, J_pde(FF, KK, kk))

    mu_f, mu_t = surfaces(gp)
    se_feas = float(np.nanmedian(SEf[feas]))                    # typical SE of a grid point's J_feat

    def optimum(mu, se):
        i = int(np.nanargmin(mu))
        thr_ = mu[i] + 2 * se
        bd = XX[mask & (np.nan_to_num(mu, nan=np.inf) <= thr_)]
        return {"F": float(FF[i]), "k": float(KK[i]), "J": float(mu[i]), "band_threshold": float(thr_),
                "band_F": [float(F0 + args.dF * bd[:, 0].min()), float(F0 + args.dF * bd[:, 0].max())],
                "band_k": [float(k0 + args.dk * bd[:, 1].min()), float(k0 + args.dk * bd[:, 1].max())]}, i, bd

    est_f, i_f, band_f = optimum(mu_f, se_feas)
    se_t = se_feas * n_crit * mu_f[int(np.nanargmin(mu_t))] / ((n_crit + 1) * np.nanmin(mu_t))   # delta method
    est_t, i_t, band = optimum(mu_t, se_t)
    from sklearn.gaussian_process import GaussianProcessRegressor
    rng = np.random.default_rng(11)
    boots_f, boots = [], []
    for _ in range(200):
        idx = rng.integers(0, Dmat.shape[1], Dmat.shape[1])
        Jb = cond_median(Dmat[:, idx])
        ok_b = feas & np.isfinite(Jb)
        gpb = GaussianProcessRegressor(gp.kernel_, alpha=(np.maximum(SEf[ok_b], 0.05) / max(np.std(Jb[ok_b]), 1e-12)) ** 2,
                                       normalize_y=True, optimizer=None).fit(X[ok_b], Jb[ok_b])
        mfb, mtb = surfaces(gpb)
        boots_f.append(XX[int(np.nanargmin(mfb))])
        boots.append(XX[int(np.nanargmin(mtb))])
    boots, boots_f = np.array(boots), np.array(boots_f)

    def pct(B):
        bF, bk = F0 + args.dF * B[:, 0], k0 + args.dk * B[:, 1]
        return ([float(np.quantile(bF, q)) for q in (0.025, 0.16, 0.5, 0.84, 0.975)],
                [float(np.quantile(bk, q)) for q in (0.025, 0.16, 0.5, 0.84, 0.975)])

    est_t["bootstrap_F"], est_t["bootstrap_k"] = pct(boots)
    est_f["bootstrap_F"], est_f["bootstrap_k"] = pct(boots_f)
    sens = {}
    for kk in kappas[1:]:
        Jk = cond_median(matrix(J_table(kk)))
        okk = feas & np.isfinite(Jk)
        mfk, mtk = surfaces(fit_gp(X[okk], Jk[okk], SEf[okk]), kk)
        sens[str(kk)] = {"F": float(FF[int(np.nanargmin(mtk))]), "k": float(KK[int(np.nanargmin(mtk))]),
                         "F_features": float(FF[int(np.nanargmin(mfk))]), "k_features": float(KK[int(np.nanargmin(mfk))]),
                         "closed_form": [cf[str(kk)]["F"], cf[str(kk)]["k"]]}
    est_t["cutoff_sensitivity"] = sens

    def at(F, k):
        x = np.array([[(F - F0) / args.dF, (k - k0) / args.dk]])
        feasible = bool(gpu.predict(x)[0] >= 0.5)
        jf = float(gp.predict(x)[0])
        jt = float(J_total(jf, J_pde(F, k)))
        return {"J_features": jf, "J_pde": float(J_pde(F, k)), "J_total": jt, "feasible": feasible,
                "in_band_features": bool(feasible and jf <= est_f["band_threshold"]),
                "in_band_total": bool(feasible and jt <= est_t["band_threshold"])}

    probes = {"given": at(*GIVEN), "closed_form": at(F0, k0), "features": at(est_f["F"], est_f["k"]),
              "estimate": at(est_t["F"], est_t["k"])}
    F_opt, k_opt = est_t["F"], est_t["k"]
    walls["surrogate"] = time.time() - t
    log(f"[5] GP kernel {gp.kernel_}; sigma_F {sig_F:.3f}, sigma_F+k {sig_S:.3f}", fh)
    log(f"    features only: F = {est_f['F']:.5f}, k = {est_f['k']:.5f}, J = {est_f['J']:.3f}, band F "
        f"{est_f['band_F'][0]:.4f}-{est_f['band_F'][1]:.4f}, k {est_f['band_k'][0]:.4f}-{est_f['band_k'][1]:.4f}", fh)
    log(f"    features + PDE: F = {F_opt:.5f}, k = {k_opt:.5f}, J = {est_t['J']:.3f}, band F "
        f"{est_t['band_F'][0]:.4f}-{est_t['band_F'][1]:.4f}, k {est_t['band_k'][0]:.4f}-{est_t['band_k'][1]:.4f}; "
        f"bootstrap 68% F {est_t['bootstrap_F'][1]:.4f}-{est_t['bootstrap_F'][3]:.4f}, "
        f"k {est_t['bootstrap_k'][1]:.4f}-{est_t['bootstrap_k'][3]:.4f}", fh)
    log(f"    probes {json.dumps(probes)}; cut-off sensitivity {json.dumps(sens)}", fh)

    # ---- 6: final steady states at the estimate, the feature-only optimum and the given values -----
    t = time.time()
    sets = {"estimate": (F_opt, k_opt), "features": (est_f["F"], est_f["k"]), "given": GIVEN}
    jobs = [(lab, float(F), float(k), s_, cfg, kh) for lab, (F, k) in sets.items() for s_ in seeds]
    finals = pool.map(job_final, jobs)
    pool.close()
    walls["final"] = time.time() - t
    for r in finals:
        r["distance"], r["per_criterion"] = feature_distance(r["features"], tgt[str(kh)])
        st = r["stability"]
        log(f"[6] {r['label']:8s} seed {r['seed']}: {r['status']}, t={r['t']:.0f}, residual {r['rel_residual']:.1e} "
            f"(steady_rel {r['steady_rel']:.1e}), max Re lambda {st['max_real'] if st else float('nan'):.2e}, "
            f"type {r['features']['type']}, n {r['features_raw']['n_spots']:.0f}, distance {r['distance']:.3f}", fh)

    def ok(r):
        return r["rel_residual"] < 1e-10 and r["stability"] is not None and r["stability"]["max_real"] < 0
    by = {lab: [r for r in finals if r["label"] == lab] for lab in sets}
    reps = {lab: min([r for r in rs if ok(r)] or rs, key=lambda r: r["distance"]) for lab, rs in by.items()}
    rep = reps["estimate"]
    np.savez(out / "final_state.npz", u=rep["u"], v=rep["v"], F=rep["F"], k=rep["k"], Du=SPOTS.Du, Dv=SPOTS.Dv,
             seed=rep["seed"], vertices=g.V, faces=g.F)
    sio.savemat(out / "solution.mat", {"Solution": {"u": rep["u"][:, None], "v": rep["v"][:, None]}})
    np.savez(out / "target_preprocessed.npz", u=up, v=vp, kappa_half=kh)

    # ---- summary ------------------------------------------------------------------------------------
    names = [n for grp in OBJECTIVE.values() for n in grp]
    summary = {
        "given_spots": {"F": GIVEN[0], "k": GIVEN[1], "Du": SPOTS.Du, "Dv": SPOTS.Dv},
        "preprocess": {"kappa_half": kh, "kappa_half_auto": kh_auto, "sigma": lp.sigma, "tau": lp.tau,
                       "v_neg_area_raw": float(g.M[vt < 0].sum() / g.area), "v_max_raw": float(vt.max()),
                       "v_max": float(vp.max()), "change_v": float(g.rms(vp - vt) / g.rms(vt - g.mean(vt))),
                       "residual_raw": [r for r in curve if not np.isfinite(r["kappa_half"])][0]["residual"],
                       "residual": cf[str(kh)]["residual"]},
        "closed_form": {"F": F0, "k": k0, "raw_clipped": raw_cf, "sigma_F_rel": sig_F, "sigma_Fk_rel": sig_S,
                        "cutoff_band": {"F": [min(r["F"] for r in cf_band), max(r["F"] for r in cf_band)],
                                        "k": [min(r["k"] for r in cf_band), max(r["k"] for r in cf_band)]}},
        "grid": {"F": list(map(float, Fg)), "k": list(map(float, kg)), "seeds": seeds, "cfg": cfg,
                 "polished_fraction": polished, "steady_fraction": steady, "status_counts": status,
                 "J": J.tolist(), "J_features": Jc.tolist(), "J_features_se": SEc.tolist(), "usable": U.tolist(),
                 "J_per_criterion": {g_: v_.tolist() for g_, v_ in per_med.items()},
                 "points": [list(p_) for p_ in pts], "per_seed": Dmat.tolist()},
        "estimate": est_t, "features_only": est_f, "probes": probes, "gp_kernel": str(gp.kernel_),
        "final": [{k_: r[k_] for k_ in ("label", "F", "k", "seed", "t", "status", "polished_in_loop", "rel_residual",
                                         "steady_rel", "stability", "distance", "per_criterion", "features",
                                         "features_raw", "wall_s")} for r in finals],
        "representative": {lab: r["seed"] for lab, r in reps.items()},
        "target_features": tgt[str(kh)], "target_features_raw": tgt_raw,
        "objective": {g_: {n: list(v_) for n, v_ in grp.items()} for g_, grp in OBJECTIVE.items()},
        "type_penalty": TYPE_PENALTY, "grid_reused": bool(args.reuse and cache.exists() and str(np.load(cache, allow_pickle=True)["key"]) == key),
        "wall_seconds": {**walls, "total": sum(walls.values()) if args.reuse else time.time() - T0},
        "workers": args.workers,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(out, summary, g, vt, vp, reps, by, curve, types, X, U, XX, mu_f, mu_t, boots, args, F0, k0,
                 names, tgt[str(kh)], per_med)
    log(f"done in {time.time() - T0:.0f}s", fh)


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def write_report(out, s, g, vt, vp, reps, by, curve, types, X, U, XX, mu_f, mu_t, boots, args, F0, k0, names,
                 tgt, per_med):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from turing_rd.viz import render_field

    kh = s["preprocess"]["kappa_half"]
    e, ef, pr = s["estimate"], s["features_only"], s["probes"]
    # cut-off curve
    c = [r for r in curve if np.isfinite(r["kappa_half"])]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([r["kappa_half"] for r in c], [r["residual"] for r in c], "k.-")
    ax[0].axhline(s["preprocess"]["residual_raw"], color="C3", ls=":", label="no low-pass")
    ax[0].axvline(kh, color="C2", label=f"chosen κ½ = {kh:g}")
    ax[0].set_xlabel("cut-off κ½ (transfer 1/2)"); ax[0].set_ylabel("steady residual / F (given Du, Dv)")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
    ax[1].plot([r["kappa_half"] for r in c], [r["F"] for r in c], "C0.-", label="F (identities)")
    ax[1].axhline(0.028, color="C0", ls=":")
    a2 = ax[1].twinx()
    a2.plot([r["kappa_half"] for r in c], [r["k"] for r in c], "C1.-", label="k (identities)")
    a2.axhline(0.062, color="C1", ls=":")
    ax[1].axvline(kh, color="C2")
    ax[1].set_xlabel("cut-off κ½"); ax[1].set_ylabel("F", color="C0"); a2.set_ylabel("k", color="C1")
    ax[1].set_title("closed form vs cut-off (dotted: given spot values)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "cutoff.png", dpi=120); plt.close(fig)

    # type map + two landscapes
    dF, dk = args.dF, args.dk
    xs, ys = np.unique(XX[:, 0]), np.unique(XX[:, 1])
    ok = U >= 0.5

    def marks(a):
        a.plot(*GIVEN[::-1], "r*", ms=14, label="given spots (truth)")
        a.plot(k0, F0, "kx", ms=9, mew=2, label="closed form")
        a.plot(ef["k"], ef["F"], "s", mfc="none", mec="orange", ms=11, mew=2, label="features only")
        a.plot(e["k"], e["F"], "o", mfc="none", mec="lime", ms=12, mew=2, label="estimate (features + PDE)")

    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    if types:
        cmap = {"uniform": "lightgray", "spots": "C0", "stripes": "C1", "maze": "C3", "other": "C4"}
        for kd in sorted({r["type"] for r in types}):
            sel = [r for r in types if r["type"] == kd]
            ax[0].scatter([r["k"] for r in sel], [r["F"] for r in sel], c=cmap.get(kd, "k"), s=55, marker="s",
                          label=kd)
        kk = np.linspace(0.04, 0.0625, 400)                    # F = 4 (F + k)^2 has real roots for k <= 1/16
        disc = np.sqrt(np.maximum(0.125 ** 2 - 0.25 * kk, 0))
        ax[0].plot(np.r_[kk, kk[::-1]], np.r_[0.125 - kk - disc, (0.125 - kk + disc)[::-1]], "k--", lw=0.8,
                   label="saddle-node F = 4(F+k)²")
        ax[0].add_patch(plt.Rectangle((min(s["grid"]["k"]), min(s["grid"]["F"])), np.ptp(s["grid"]["k"]),
                                      np.ptp(s["grid"]["F"]), fill=False, ec="k", lw=1.2))
    marks(ax[0])
    ax[0].set_xlabel("k"); ax[0].set_ylabel("F"); ax[0].set_ylim(0.005, 0.075)
    ax[0].set_title("type at t = 5000 (one initial state); box: fine grid", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper left")
    for a, mu, est_, title in ((ax[1], mu_f, ef, "features only: J_feat (steady spot seeds)"),
                               (ax[2], mu_t, e, "features + PDE residual: J_total")):
        M = mu.reshape(len(xs), len(ys))
        im = a.contourf(k0 + dk * ys, F0 + dF * xs, M, levels=20, cmap="viridis_r")
        a.contour(k0 + dk * ys, F0 + dF * xs, M, levels=[est_["band_threshold"]], colors="w", linewidths=1.5)
        a.scatter(k0 + dk * X[ok, 1], F0 + dF * X[ok, 0], c="k", s=12)
        a.scatter(k0 + dk * X[~ok, 1], F0 + dF * X[~ok, 0], marker="x", c="gray", s=30,
                  label="< half of the seeds steady spots")
        if a is ax[2]:
            a.scatter(k0 + dk * boots[:, 1], F0 + dF * boots[:, 0], s=6, c="w", alpha=0.5, label="bootstrap minima")
        marks(a)
        fig.colorbar(im, ax=a, label="GP mean (white line: band)")
        a.set_xlim(k0 + dk * (ys.min() - 0.3), k0 + dk * (ys.max() + 0.3))
        a.set_ylim(F0 + dF * (xs.min() - 0.3), F0 + dF * (xs.max() + 0.3))
        a.set_xlabel("k"); a.set_ylabel("F"); a.set_title(title, fontsize=10); a.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "landscape.png", dpi=120); plt.close(fig)

    fig, axs = plt.subplots(1, len(per_med), figsize=(5 * len(per_med), 4))
    for a, (name, vals) in zip(np.atleast_1d(axs), per_med.items()):
        sc = a.scatter(k0 + dk * X[ok, 1], F0 + dF * X[ok, 0], c=vals[ok], cmap="viridis_r", s=160, marker="s")
        a.scatter(k0 + dk * X[~ok, 1], F0 + dF * X[~ok, 0], marker="x", c="gray", s=30)
        a.plot(*GIVEN[::-1], "r*", ms=12)
        a.plot(e["k"], e["F"], "o", mfc="none", mec="lime", ms=12, mew=2)
        fig.colorbar(sc, ax=a)
        a.set_title(f"criterion '{name}' (median over seeds, tolerance units)", fontsize=9)
        a.set_xlabel("k"); a.set_ylabel("F")
    fig.tight_layout(); fig.savefig(out / "criteria.png", dpi=110); plt.close(fig)

    tmp = out / "_tmp.png"
    imgs = []
    panels = [(vt, "target Solution.mat (raw)"), (vp, f"target preprocessed (low-pass κ½ = {kh:g}, clipped)")]
    for lab, name in (("estimate", "estimate"), ("given", "given (truth)")):
        r = reps[lab]
        panels.append((r["v"], f"{name} F={r['F']:.4f}, k={r['k']:.4f}: steady state (seed {r['seed']}), "
                                f"residual {r['rel_residual']:.0e}, n={r['features_raw']['n_spots']:.0f}"))
    for f, title in panels:
        render_field(g, f, tmp, title=title, vmin=0, vmax=0.5)
        imgs.append(plt.imread(tmp))
    tmp.unlink()
    fig, axs = plt.subplots(len(imgs), 1, figsize=(12, 3.6 * len(imgs)))
    for a, im_ in zip(axs, imgs):
        a.imshow(im_); a.axis("off")
    fig.tight_layout(); fig.savefig(out / "final_vs_target.png", dpi=100); plt.close(fig)

    def row(name, F, k, p_):
        return (f"| {name} | {F:.5f} | {k:.5f} | {F / 0.028 - 1:+.1%} | {k / 0.062 - 1:+.1%} | "
                f"{'是' if p_['feasible'] else '否'} | {p_['J_features']:.2f} | {p_['J_pde']:.2f} | {p_['J_total']:.2f} |")

    L = ["# Solution.mat 反演（只反演 F、k；Du、Dv 取题给值）\n",
         f"一键命令：`python scripts/invert_target.py`；总用时 {s['wall_seconds']['total'] / 60:.1f} 分钟"
         + ("（网格取自缓存，用时为原始计算时间，其余阶段为本次重算）" if s.get("grid_reused") else "") +
         f"（Mac CPU，{s['workers']} 进程）。各阶段：" +
         "，".join(f"{k_} {v_:.0f} s" for k_, v_ in s["wall_seconds"].items() if k_ != "total") + "。\n",
         "## 结果\n",
         "| 参数组 | F | k | F 误差 | k 误差 | 稳态斑点区内 | J_feat | J_pde | J_total |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
         row("题给斑点参数（真值）", *GIVEN, pr["given"]),
         row("积分恒等式（闭式初值）", F0, k0, pr["closed_form"]),
         row("仅特征匹配（GP 极小）", ef["F"], ef["k"], pr["features"]),
         row("**特征 + PDE 残差（最终）**", e["F"], e["k"], pr["estimate"]), "",
         f"最终估计的不确定带（J_total ≤ 极小 + 2 个标准误 = {e['band_threshold']:.3f}）：F ∈ [{e['band_F'][0]:.4f}, "
         f"{e['band_F'][1]:.4f}]，k ∈ [{e['band_k'][0]:.4f}, {e['band_k'][1]:.4f}]；种子自助（200 次）68% 区间："
         f"F [{e['bootstrap_F'][1]:.4f}, {e['bootstrap_F'][3]:.4f}]，k [{e['bootstrap_k'][1]:.4f}, {e['bootstrap_k'][3]:.4f}]。"
         f"仅特征的带：F ∈ [{ef['band_F'][0]:.4f}, {ef['band_F'][1]:.4f}]，k ∈ [{ef['band_k'][0]:.4f}, {ef['band_k'][1]:.4f}]。"
         "截断敏感性（κ½ 取 0.8 / 1.25 倍）：" +
         "；".join(f"κ½ = {k_}：F {v_['F']:.4f}，k {v_['k']:.4f}（仅特征 {v_['F_features']:.4f} / {v_['k_features']:.4f}）"
                  for k_, v_ in e["cutoff_sensitivity"].items()) + "。\n",
         f"真值在带内：最终 {'是' if pr['given']['in_band_total'] else '否'}，仅特征 "
         f"{'是' if pr['given']['in_band_features'] else '否'}。\n",
         "## 网格\n",
         f"{len(s['grid']['F'])} × {len(s['grid']['k'])} 个 (F, k)，每点 {len(s['grid']['seeds'])} 个固定随机初值（共同随机数），"
         f"积分上限 t = {s['grid']['cfg']['t_max']:g}；PTC 精修到 1e-10 的占 {s['grid']['polished_fraction']:.0%}，"
         f"按稳态判据算作稳态的占 {s['grid']['steady_fraction']:.0%}。表中为 J_feat（稳态斑点种子的中位数）/ 稳态斑点种子比例；"
         "比例 < 0.5 的点不进 GP。\n",
         "| F \\ k | " + " | ".join(f"{k_:.4f}" for k_ in s["grid"]["k"]) + " |",
         "| --- " * (len(s["grid"]["k"]) + 1) + "|"]
    idx = {(round(a_, 7), round(b_, 7)): i for i, (a_, b_) in enumerate(s["grid"]["points"])}
    for F_ in sorted(s["grid"]["F"], reverse=True):
        cells = []
        for k_ in s["grid"]["k"]:
            i = idx[(round(F_, 7), round(k_, 7))]
            jf = s["grid"]["J_features"][i]
            cells.append(f"{jf:.2f} / {s['grid']['usable'][i]:.2f}" if np.isfinite(jf) else f"— / {s['grid']['usable'][i]:.2f}")
        L.append(f"| {F_:.4f} | " + " | ".join(cells) + " |")
    L += ["", "## 稳态、残差与稳定性\n",
          "| 参数组 | 种子 | 终止 t | 相对残差 | steady_rel | max Re λ | 类型 | 数量 | 距离 J_feat |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    label = {"estimate": "最终估计", "features": "仅特征", "given": "题给"}
    for lab, rs in by.items():
        for r in rs:
            st = r["stability"]
            L.append(f"| {label[lab]} | {r['seed']} | {r['t']:.0f} | {r['rel_residual']:.1e} | {r['steady_rel']:.1e} | "
                     f"{st['max_real'] if st else float('nan'):.1e} | {r['features']['type']} | "
                     f"{r['features_raw']['n_spots']:.0f} | {r['distance']:.2f} |")
    L += ["", "## 特征对照（同一测量算子：低通 + 截断；各参数组为种子中位数）\n",
          "| 特征 | 容差 | 目标 | 最终估计 | 仅特征 | 题给参数 |", "| --- | --- | --- | --- | --- | --- |"]
    tol = {n: v_ for grp in OBJECTIVE.values() for n, v_ in grp.items()}
    for n in names + ["n_spots", "spot_irregular_frac", "spot_size_outlier_frac", "stripe_angle_deg"]:
        tl = tol.get(n)
        tl_s = (f"±{tl[1]:.0%}" if tl[0] == "log" else f"±{tl[1]:g}") if tl else "（不计分）"
        L.append(f"| {n} | {tl_s} | {tgt[n]:.3f} | " +
                 " | ".join(f"{np.median([r['features'][n] for r in by[lab]]):.3f}" for lab in ("estimate", "features", "given"))
                 + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
