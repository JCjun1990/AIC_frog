"""Steady-state inversion with F and k fixed by the steady integral identities.

Every steady state of the Gray-Scott system on a closed surface satisfies (integrate each equation;
the diffusion terms vanish):

    F = int u v^2 / int (1 - u),        F + k = int u v^2 / int v

so under the steady-state hypothesis F and k follow from the data alone, independently of the
discretisation of the Laplacian. Here they are computed on the admissible target (h-scale smoothing,
u in [0, 1], v >= 0) and held fixed, and only (Du, Dv) are fitted by minimising the drift of
scripts/stationarity_inversion.py with gradients through the solver.

Writes results/stationarity/constrained.json and appends a section to results/stationarity/summary.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import stationarity_inversion as SI  # noqa: E402
from turing_rd import Geometry, load_frog  # noqa: E402


def main(iters: int = 60, lr: float = 0.03):
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=ROOT / "results" / "cache")
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    u0, v0 = SI.admissible(g, S["u"][0, 0].ravel(), S["v"][0, 0].ravel())
    uv2 = g.M @ (u0 * v0 * v0)
    F = float(uv2 / (g.M @ (1 - u0)))
    k = float(uv2 / (g.M @ v0)) - F
    print(f"integral identities on the admissible target: F = {F:.5f}, k = {k:.5f}", flush=True)
    drift = SI.Drift(g, u0, v0)
    starts = {"stated D": (0.0, 0.0), "Dv = Du/3": (0.0, np.log(2 / 3)), "larger Du": (np.log(1.3), np.log(1.3))}
    best = {}
    for tag, (a, b) in starts.items():
        z = torch.tensor([a, b], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        rec = (np.inf, None)
        t0 = time.time()
        for it in range(iters):
            opt.zero_grad()
            P = {"F": torch.tensor(F, dtype=torch.float64), "k": torch.tensor(k, dtype=torch.float64),
                 "Du": SI.REF["Du"] * torch.exp(z[0]), "Dv": SI.REF["Dv"] * torch.exp(z[1])}
            loss, _ = drift(P)
            loss.backward()
            opt.step()
            with torch.no_grad():
                z.clamp_(-2.5, 2.5)
            Du, Dv = float(P["Du"].detach()), float(P["Dv"].detach())
            if float(loss.detach()) < rec[0]:
                rec = (float(loss.detach()), {"F": F, "k": k, "Du": Du, "Dv": Dv})
            if it % 10 == 0 or it == iters - 1:
                print(f"[{tag}] it {it:2d} drift={float(loss.detach()):.4f} Du={Du:.3g} Dv={Dv:.3g} Dv/Du={Dv / Du:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        best[tag] = rec
    tag = min(best, key=lambda t: best[t][0])
    out = {"F": F, "k": k, "best_start": tag, "drift": best[tag][0], **best[tag][1],
           "Dv/Du": best[tag][1]["Dv"] / best[tag][1]["Du"],
           "all": {t: {"drift": b[0], **b[1]} for t, b in best.items()}}
    (SI.OUT / "constrained.json").write_text(json.dumps(out, indent=2))
    with open(SI.OUT / "summary.md", "a") as fh:
        fh.write(f"\n## F、k 由稳态积分恒等式固定（F = {F:.4f}, k = {k:.4f}），只优化扩散系数\n\n"
                 "| 起点 | 漂移 D | Du | Dv | Dv/Du |\n| --- | --- | --- | --- | --- |\n")
        for t, b in best.items():
            fh.write(f"| {t} | {b[0]:.4f} | {b[1]['Du']:.3g} | {b[1]['Dv']:.3g} | {b[1]['Dv'] / b[1]['Du']:.3f} |\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
