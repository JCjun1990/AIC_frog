"""Stationarity inversion: under which (F, k, Du, Dv) is Solution.mat closest to a steady state?

Start from the target made admissible (light h-scale smoothing, then u clipped to [0, 1] and v to
>= 0, the bounds every Gray-Scott steady state obeys), integrate with the differentiable solver and
minimise the drift

    D(theta) = mean over checkpoints t of  ||v(t) - v0||^2 / ||v0 - mean||^2 + same for u

by gradient descent (Adam through the IMEX solver: gradients of the linear solves come from
solver._SPDSolve). A stable steady state that is only lightly perturbed relaxes back and gives a
small D; a non-steady field drifts on the reaction time scale.

Reference ("noise floor"): our own converged steady state for the stated spot parameters with the
target's small-scale component added on top, evaluated at those parameters.

Writes results/stationarity/summary.md, history.csv, optimum.json and drift.png.

    python scripts/stationarity_inversion.py
"""
from __future__ import annotations

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
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, GrayScottSolver, load_frog  # noqa: E402

OUT = ROOT / "results" / "stationarity"
REF = {"F": 0.028, "k": 0.062, "Du": 2.78e-4, "Dv": 1.39e-4}
CHECKPOINTS = (25, 50, 100, 150, 200)


def admissible(g, u, v, c=1.0):
    A = (sp.diags(g.M) - c * g.h_med ** 2 * g.L).tocsc()
    us = spla.spsolve(A, g.M * u) if c else u
    vs = spla.spsolve(A, g.M * v) if c else v
    return np.clip(us, 0, 1), np.maximum(vs, 0)


class Drift:
    def __init__(self, g, u0, v0, dt=1.0):
        self.g, self.dt = g, dt
        self.solver = GrayScottSolver(g, dt=dt, scheme="sbdf2", backend="lu")
        self.u0 = torch.as_tensor(u0)
        self.v0 = torch.as_tensor(v0)
        M = torch.as_tensor(g.M)
        self.M = M
        self.nu = float(M @ (self.u0 - (M @ self.u0) / M.sum()) ** 2)
        self.nv = float(M @ (self.v0 - (M @ self.v0) / M.sum()) ** 2)

    def __call__(self, P, checkpoints=CHECKPOINTS, keep=False):
        u, v, t, loss, trace = self.u0, self.v0, 0, 0.0, []
        for tc in checkpoints:
            n = int(round((tc - t) / self.dt))
            u, v = self.solver.rollout(u, v, P, n)
            t = tc
            term =(self.M @ (u - self.u0) ** 2) / self.nu + (self.M @ (v - self.v0) ** 2) / self.nv
            loss = loss + term
            trace.append(float(term.detach()))
        loss = loss / len(checkpoints)
        return (loss, trace, (u.detach().numpy(), v.detach().numpy())) if keep else (loss, trace)


def params_from(z):
    return {"F": 0.03 * z[0], "k": 0.06 * z[1], "Du": REF["Du"] * torch.exp(z[2]), "Dv": REF["Dv"] * torch.exp(z[3])}


def as_floats(P):
    return {k: float(v) for k, v in P.items()}


def optimise(drift, start, iters=80, lr=0.02, log=None, tag=""):
    z0 = torch.tensor([start["F"] / 0.03, start["k"] / 0.06, np.log(start["Du"] / REF["Du"]),
                       np.log(start["Dv"] / REF["Dv"])], dtype=torch.float64)
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    best = (np.inf, None)
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        P = params_from(z)
        loss, trace = drift(P)
        loss.backward()
        opt.step()
        with torch.no_grad():                       # keep F, k positive and inside the pattern region
            z[0].clamp_(0.2, 3.0); z[1].clamp_(0.5, 1.4); z[2].clamp_(-2.5, 2.5); z[3].clamp_(-3.0, 2.5)
        Pf = as_floats(P)
        if float(loss) < best[0]:
            best = (float(loss), Pf)
        row = {"start": tag, "iter": it, "loss": float(loss), **Pf, "Dv/Du": Pf["Dv"] / Pf["Du"],
               "wall_s": time.time() - t0}
        if log is not None:
            log.append(row)
        if it % 5 == 0 or it == iters - 1:
            print(f"[{tag}] it {it:3d} drift={float(loss):.4f}  F={Pf['F']:.4f} k={Pf['k']:.4f} "
                  f"Du={Pf['Du']:.3g} Dv={Pf['Dv']:.3g} (Dv/Du={Pf['Dv'] / Pf['Du']:.3f})  ({time.time() - t0:.0f}s)",
                  flush=True)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    ut, vt = S["u"][0, 0].ravel(), S["v"][0, 0].ravel()
    u0, v0 = admissible(g, ut, vt)
    drift = Drift(g, u0, v0)

    # noise floor: our steady state at the stated parameters + the target's small-scale component
    b = np.load(ROOT / "results" / "baseline" / "frog_spots" / "final.npz")
    A = (sp.diags(g.M) - g.h_med ** 2 * g.L).tocsc()
    hu = ut - spla.spsolve(A, g.M * ut)
    hv = vt - spla.spsolve(A, g.M * vt)
    floor = {}
    with torch.no_grad():
        for name, (uu, vv) in {"steady state, unperturbed": (b["u"], b["v"]),
                               "steady state + target small-scale part": admissible(g, b["u"] + hu, b["v"] + hv, c=0)}.items():
            floor[name] = float(Drift(g, uu, vv)(REF)[0])
            print(f"noise floor — {name}: drift {floor[name]:.4f}", flush=True)
        given = float(drift(REF)[0])
        print(f"target at the stated spot parameters: drift {given:.4f}", flush=True)

    starts = {
        "stated spot parameters": dict(REF),
        "integral identities + stated D": {"F": 0.0290, "k": 0.0611, "Du": 2.78e-4, "Dv": 1.39e-4},
        "snapshot inversion (CMA-ES)": {"F": 0.0337, "k": 0.0608, "Du": 2.53e-4, "Dv": 8.64e-5},
    }
    log, bests = [], {}
    for tag, st in starts.items():
        bests[tag] = optimise(drift, st, log=log, tag=tag)
    with open(OUT / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    tag_opt = min(bests, key=lambda t: bests[t][0])
    loss_opt, P_opt = bests[tag_opt]

    # longer drift curves from the target: stated parameters vs optimum
    def wcorr(a, b):
        a = a - g.mean(a); b = b - g.mean(b)
        return float((g.M @ (a * b)) / np.sqrt((g.M @ (a * a)) * (g.M @ (b * b))))
    curves = {}
    with torch.no_grad():
        for name, P in {"stated spot parameters": REF, f"stationarity optimum": P_opt}.items():
            ts, cs = [], []
            u, v, t = torch.as_tensor(u0), torch.as_tensor(v0), 0
            for tc in (10, 25, 50, 100, 200, 400, 800, 1500, 3000):
                u, v = drift.solver.rollout(u, v, P, tc - t)
                t = tc
                ts.append(tc); cs.append(wcorr(v.numpy(), vt))
            curves[name] = (ts, cs)
            print(f"{name}: corr(v(t), target) " + "  ".join(f"t={a}:{c:.3f}" for a, c in zip(ts, cs)), flush=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (ts, cs) in curves.items():
        ax.semilogx(ts, cs, "o-", label=name)
    ax.set_xlabel("t (starting from the admissible target)"); ax.set_ylabel("pointwise correlation with target")
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Does the target stay put?")
    fig.tight_layout(); fig.savefig(OUT / "drift.png", dpi=120)

    result = {"noise_floor": floor, "drift_at_stated": given, "best_start": tag_opt, "drift_opt": loss_opt,
              "optimum": P_opt, "Dv/Du": P_opt["Dv"] / P_opt["Du"], "all_starts": {t: {"drift": b[0], **b[1]} for t, b in bests.items()},
              "corr_curves": {k: {"t": v[0], "corr": v[1]} for k, v in curves.items()}}
    (OUT / "optimum.json").write_text(json.dumps(result, indent=2))
    lines = ["# 稳态反演（漂移最小化）\n",
             f"漂移 D = 检查点 t ∈ {list(CHECKPOINTS)} 上 u、v 相对初值的归一化平方偏差均值；初值为目标去掉不可能取值后的场。\n",
             "| 情形 | 漂移 D |", "| --- | --- |"]
    for k, v in floor.items():
        lines.append(f"| 参照：{k}，题给参数 | {v:.4f} |")
    lines.append(f"| 目标，题给斑点参数 | {given:.4f} |")
    for t, b in bests.items():
        lines.append(f"| 目标，从「{t}」出发优化后 | {b[0]:.4f}（F={b[1]['F']:.4f}, k={b[1]['k']:.4f}, Du={b[1]['Du']:.3g}, "
                     f"Dv={b[1]['Dv']:.3g}, Dv/Du={b[1]['Dv'] / b[1]['Du']:.3f}） |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
