"""Trajectory scan over (F, k) for one (Du, Dv): pointwise and statistical match to Solution.mat.

All (F, k) on a grid are integrated together on the frog (SBDF2, shared LU factorisations, one
column per (F, k)). Every `--log-every` steps each live column is compared with the target:

    pointwise    area-weighted correlation of v (and u) with the target; best value and time kept
    --stats      pattern statistics (type, spot count, wavelength, area fraction, compactness, v peak)
                 and their distance to the target's (diagnostics.feature_distance); every time window
                 with shape distance <= 1 is recorded, plus the best-matching state

Columns that die (max v < 1e-4), blow up, or stop changing are retired early (their final
statistics are recorded). Writes results/scan/<tag>/: scan.csv, best_states.npz, heatmap.png, and
with --stats also stats_scan.csv, timeseries.npz, stats_best_states.npz.

    python scripts/trajectory_scan.py --du-scale 1.0 --ratio 0.5
    python scripts/trajectory_scan.py --du-scale 0.7 --ratio 0.5 --ic seeded --stats
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, load_frog  # noqa: E402
from turing_rd.diagnostics import feature_distance, pattern_stats  # noqa: E402
from turing_rd.initial import perturbed_ic, suggested_ic  # noqa: E402

DU0 = 2.78e-4
STAT_KEYS = ("wavelength", "n_components", "area_fraction", "compactness", "v_max")


def windows(times, ok):
    """Consecutive runs of True in ok -> list of [t_start, t_end]."""
    out, start = [], None
    for t, o in zip(times, ok):
        if o and start is None:
            start = t
        if not o and start is not None:
            out.append([start, prev])
            start = None
        prev = t
    if start is not None:
        out.append([start, prev])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--du-scale", type=float, default=1.0, help="Du = scale * 2.78e-4")
    ap.add_argument("--ratio", type=float, default=0.5, help="Dv / Du")
    ap.add_argument("--F", type=float, nargs=3, default=[0.010, 0.080, 15], metavar=("MIN", "MAX", "N"))
    ap.add_argument("--k", type=float, nargs=3, default=[0.040, 0.075, 15], metavar=("MIN", "MAX", "N"))
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=4000)
    ap.add_argument("--log-every", type=int, default=20, help="steps between comparisons")
    ap.add_argument("--laplacian", default="robust")
    ap.add_argument("--ic", choices=["suggested", "seeded"], default="suggested",
                    help="organisers' formula as written, or the baseline protocol (z-normalised + seeds)")
    ap.add_argument("--normalize-z", action="store_true", help="(suggested IC) scale z by the half-height")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="also track pattern statistics")
    ap.add_argument("--keep", type=int, default=12, help="best states saved")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "scan"))
    args = ap.parse_args(argv)

    Du, Dv = args.du_scale * DU0, args.du_scale * DU0 * args.ratio
    ic_tag = "_seeded" if args.ic == "seeded" else ("_nz" if args.normalize_z else "")
    tag = args.tag or f"Du{args.du_scale:g}_r{args.ratio:g}{ic_tag}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    g = Geometry(load_frog(ROOT / "frog.mat"), laplacian=args.laplacian, cache_dir=None)
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ut, vt = S["u"][0, 0].ravel(), S["v"][0, 0].ravel()
    target_stats = pattern_stats(g, vt)
    M, A = g.M, g.area

    def centred(X):
        return X - (M @ X) / A

    tu, tv = centred(ut), centred(vt)
    nu, nv = np.sqrt(M @ (tu * tu)), np.sqrt(M @ (tv * tv))

    def corr(X, tc, tn):
        Xc = centred(X)
        return (M @ (Xc * tc[:, None])) / (np.sqrt(M @ (Xc * Xc)) * tn + 1e-300)

    Fs = np.linspace(args.F[0], args.F[1], int(args.F[2]))
    ks = np.linspace(args.k[0], args.k[1], int(args.k[2]))
    FF, KK = (x.ravel() for x in np.meshgrid(Fs, ks, indexing="ij"))
    B = len(FF)
    if args.ic == "seeded":
        u0, v0 = perturbed_ic(g, seed=args.seed, mode="seeded", normalize_z=True)
    else:
        u0, v0 = suggested_ic(g, normalize_z=args.normalize_z)

    dt = args.dt
    Md = sp.diags(M)
    lu1u, lu1v = (spla.splu((Md - dt * D * g.L).tocsc()) for D in (Du, Dv))
    lu2u, lu2v = (spla.splu((1.5 * Md - dt * D * g.L).tocsc()) for D in (Du, Dv))

    def react(u, v, F, k):
        r = u * v * v
        return -r + F * (1 - u), r - (F + k) * v

    alive = np.arange(B)
    F, k = FF.copy(), KK.copy()
    u = np.repeat(u0[:, None], B, 1)
    v = np.repeat(v0[:, None], B, 1)
    fu, fv = react(u, v, F, k)
    up, vp = u, v
    u = lu1u.solve(M[:, None] * (u + dt * fu))
    v = lu1v.solve(M[:, None] * (v + dt * fv))
    fup, fvp = fu, fv

    best_cv = np.full(B, -2.0); best_cu = np.full(B, np.nan); best_t = np.zeros(B)
    best_rel = np.full(B, np.nan); status = np.array(["running"] * B, dtype=object)
    t_end = np.zeros(B); vmax_end = np.zeros(B)
    best_u = np.zeros((g.n, B)); best_v = np.zeros((g.n, B))
    n_logs = int(round(args.t_max / dt)) // args.log_every
    log_t = np.arange(1, n_logs + 1) * args.log_every * dt
    if args.stats:
        ts = {key: np.full((B, n_logs), np.nan) for key in STAT_KEYS + ("shape_d", "peak_d")}
        ts["is_spots"] = np.zeros((B, n_logs), dtype=bool)
        sbest_d = np.full(B, np.inf); sbest_t = np.zeros(B); sbest = [None] * B
        sbest_u = np.zeros((g.n, B)); sbest_v = np.zeros((g.n, B))
        final_stats = [None] * B

    def record_stats(cols, X, li):
        for j, c in enumerate(cols):
            s = pattern_stats(g, X[:, j])
            d, pk = feature_distance(s, target_stats)
            for key in STAT_KEYS:
                ts[key][c, li] = s.get(key, 0) or 0
            ts["shape_d"][c, li], ts["peak_d"][c, li] = d, pk
            ts["is_spots"][c, li] = s["type"] == "spots"
            if d < sbest_d[c]:
                sbest_d[c], sbest_t[c], sbest[c] = d, log_t[li], s
                sbest_u[:, c], sbest_v[:, c] = u[:, j], X[:, j]

    v_last = v.copy()
    n_steps = n_logs * args.log_every
    t0 = time.time()
    for step in range(2, n_steps + 1):
        fu, fv = react(u, v, F, k)
        un = lu2u.solve(M[:, None] * (2 * u - 0.5 * up + dt * (2 * fu - fup)))
        vn = lu2v.solve(M[:, None] * (2 * v - 0.5 * vp + dt * (2 * fv - fvp)))
        up, vp, u, v, fup, fvp = u, v, un, vn, fu, fv
        if step % args.log_every:
            continue
        li = step // args.log_every - 1
        t = step * dt
        cv, cu = corr(v, tv, nv), corr(u, tu, nu)
        better = cv > best_cv[alive]
        if better.any():
            idx = alive[better]
            best_cv[idx], best_cu[idx], best_t[idx] = cv[better], cu[better], t
            best_rel[idx] = np.sqrt(M @ ((v[:, better] - vt[:, None]) ** 2) / A) / np.sqrt(M @ (vt * vt) / A)
            best_u[:, idx], best_v[:, idx] = u[:, better], v[:, better]
        if args.stats:
            record_stats(alive, v, li)
        vmax = np.nanmax(np.abs(v), axis=0)
        rate = np.sqrt(M @ ((v - v_last) ** 2) / A) / (args.log_every * dt)
        dead = vmax < 1e-4
        blown = ~np.isfinite(vmax) | (vmax > 10)
        steady = (rate < 1e-7) & ~dead & (t > 200)
        retire = dead | blown | steady
        if retire.any():
            idx = alive[retire]
            status[idx] = np.where(dead[retire], "dead", np.where(blown[retire], "blown", "steady"))
            t_end[idx], vmax_end[idx] = t, vmax[retire]
            if args.stats:
                for j, c in zip(np.where(retire)[0], idx):
                    final_stats[c] = pattern_stats(g, v[:, j])
                    if status[c] == "steady":        # a steady state keeps its statistics to t_max
                        for key in STAT_KEYS + ("shape_d", "peak_d", "is_spots"):
                            ts[key][c, li + 1:] = ts[key][c, li]
            keep = ~retire
            alive, F, k = alive[keep], F[keep], k[keep]
            u, v, up, vp, fup, fvp = (x[:, keep] for x in (u, v, up, vp, fup, fvp))
            v_last = v.copy()
            if alive.size == 0:
                break
        else:
            v_last = v.copy()
        if step % (args.log_every * 25) == 0:
            top = np.argsort(-best_cv)[:2]
            msg = f"[{tag}] t={t:6.0f} alive={alive.size:3d}/{B} best corr_v=" + ", ".join(
                f"{best_cv[i]:.3f}(F={FF[i]:.4f},k={KK[i]:.4f})" for i in top)
            if args.stats:
                i = int(np.argmin(sbest_d))
                msg += (f" | best shape_d={sbest_d[i]:.2f}(F={FF[i]:.4f},k={KK[i]:.4f},t={sbest_t[i]:.0f}) "
                        f"n(shape_d<=1)={(sbest_d <= 1).sum()}")
            print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
    t_end[alive] = n_steps * dt
    if alive.size:
        vmax_end[alive] = np.nanmax(np.abs(v), axis=0)
        if args.stats:
            for j, c in enumerate(alive):
                final_stats[c] = pattern_stats(g, v[:, j])

    rows = [{"F": FF[i], "k": KK[i], "Du": Du, "Dv": Dv, "best_corr_v": best_cv[i], "best_corr_u": best_cu[i],
             "t_best": best_t[i], "rel_l2_v": best_rel[i], "status": status[i], "t_end": t_end[i],
             "vmax_end": vmax_end[i]} for i in range(B)]
    with open(out / "scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    top = np.argsort(-best_cv)[:args.keep]
    np.savez(out / "best_states.npz", F=FF[top], k=KK[top], t=best_t[top], corr_v=best_cv[top],
             u=best_u[:, top], v=best_v[:, top], Du=Du, Dv=Dv)
    meta = {**vars(args), "Du": Du, "Dv": Dv, "wall_seconds": time.time() - t0, "target_stats": target_stats}

    if args.stats:
        srows = []
        for i in range(B):
            ok = ts["shape_d"][i] <= 1.0
            win = windows(log_t, ok)
            okp = ok & (ts["peak_d"][i] <= 1.0)
            fs, sb = final_stats[i] or {}, sbest[i] or {}
            fd = feature_distance(fs, target_stats)[0] if fs else np.inf
            srows.append({
                "F": FF[i], "k": KK[i], "Du": Du, "Dv": Dv, "ic": args.ic, "status": status[i], "t_end": t_end[i],
                "best_shape_d": sbest_d[i], "t_best": sbest_t[i],
                **{f"best_{key}": sb.get(key, "") for key in STAT_KEYS},
                "match_time": float(ok.sum() * args.log_every * dt),
                "match_first": win[0][0] if win else "", "match_last": win[-1][1] if win else "",
                "match_windows": json.dumps(win), "match_time_with_peak": float(okp.sum() * args.log_every * dt),
                "final_type": fs.get("type", ""), "final_n": fs.get("n_components", ""),
                "final_wavelength": fs.get("wavelength", ""), "final_area_fraction": fs.get("area_fraction", ""),
                "final_v_max": fs.get("v_max", ""), "final_shape_d": fd})
        with open(out / "stats_scan.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(srows[0]))
            w.writeheader()
            w.writerows(srows)
        np.savez_compressed(out / "timeseries.npz", t=log_t, F=FF, k=KK, **ts)
        topS = np.argsort(sbest_d)[:args.keep]
        np.savez(out / "stats_best_states.npz", F=FF[topS], k=KK[topS], t=sbest_t[topS], shape_d=sbest_d[topS],
                 u=sbest_u[:, topS], v=sbest_v[:, topS], Du=Du, Dv=Dv)
        i = topS[0]
        meta["stats_summary"] = {"n_with_match": int((sbest_d <= 1).sum()), "best_shape_d": float(sbest_d[i]),
                                 "best_F": float(FF[i]), "best_k": float(KK[i]), "best_t": float(sbest_t[i])}

    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    fig, ax = plt.subplots(1, 2 if args.stats else 1, figsize=(11 if args.stats else 6, 4.6), squeeze=False)
    ext = [Fs[0], Fs[-1], ks[0], ks[-1]]
    im = ax[0, 0].imshow(best_cv.reshape(len(Fs), len(ks)).T, origin="lower", aspect="auto", cmap="magma",
                         extent=ext, vmin=0, vmax=max(0.3, best_cv.max()))
    fig.colorbar(im, ax=ax[0, 0], label="best pointwise corr(v, target)")
    ax[0, 0].set_xlabel("F"); ax[0, 0].set_ylabel("k")
    if args.stats:
        im2 = ax[0, 1].imshow(np.log10(np.clip(sbest_d, 0.1, 100)).reshape(len(Fs), len(ks)).T, origin="lower",
                              aspect="auto", cmap="viridis_r", extent=ext, vmin=-1, vmax=2)
        fig.colorbar(im2, ax=ax[0, 1], label="log10 best statistical distance (≤ 0 = match)")
        ax[0, 1].set_xlabel("F"); ax[0, 1].set_ylabel("k")
    fig.suptitle(f"{tag}: Du={Du:.3g}, Dv={Dv:.3g}, IC={args.ic}, t ≤ {args.t_max:g}")
    fig.tight_layout(); fig.savefig(out / "heatmap.png", dpi=110)
    i = top[0]
    msg = f"[{tag}] DONE best corr_v={best_cv[i]:.3f} at F={FF[i]:.4f}, k={KK[i]:.4f}, t={best_t[i]:.0f}"
    if args.stats:
        msg += " | " + json.dumps(meta["stats_summary"])
    print(msg + "; statuses: " + json.dumps({s: int((status == s).sum()) for s in set(status)})
          + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
