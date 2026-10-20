"""Initial-condition study behind the baseline protocol (evidence for the report).

For each surface x parameter set x IC variant x 3 random seeds, integrate to t = 3000 and record
the pattern type. Shows that the organisers' suggested IC alone decays in the spot regime, that
small additive noise is marginal, and that z-normalised seeding is robust on both surfaces.
Writes results/verification/ic_sensitivity.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, GrayScottSolver, SPOTS, STRIPES, icosphere, load_frog  # noqa: E402
from turing_rd.diagnostics import pattern_stats  # noqa: E402
from turing_rd.initial import perturbed_ic, suggested_ic  # noqa: E402

VARIANTS = [  # (label, mode, normalize_z)
    ("suggested (no noise)", "suggested", False),
    ("suggested + 5% noise", "additive", False),
    ("suggested + seeds", "seeded", False),
    ("z-normalised + seeds", "seeded", True),
    ("Pearson background + seeds", "pearson", False),
]


def main(t_end: int = 3000):
    out = ROOT / "results" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    surfaces = {"sphere_L6": icosphere(6), "frog": load_frog(ROOT / "frog.mat")}
    rows = []
    for sname, mesh in surfaces.items():
        g = Geometry(mesh, cache_dir=None)
        for pname, p in (("spots", SPOTS), ("stripes", STRIPES)):
            for label, mode, nz in VARIANTS:
                seeds = (0,) if mode == "suggested" else (0, 1, 2)
                for seed in seeds:
                    if mode == "suggested":
                        u0, v0 = suggested_ic(g, normalize_z=nz)
                    else:
                        u0, v0 = perturbed_ic(g, eps=0.05, seed=seed, mode=mode, normalize_z=nz)
                    with torch.no_grad():
                        u, v = GrayScottSolver(g, dt=1.0).rollout(u0, v0, p, t_end)
                    s = pattern_stats(g, v.numpy())
                    row = {"surface": sname, "params": pname, "ic": label, "seed": seed, "t": t_end,
                           "type": s["type"], "n_components": s.get("n_components", 0),
                           "area_fraction": round(s.get("area_fraction", 0.0), 3),
                           "wavelength": round(s["wavelength"], 3) if s["wavelength"] != float("inf") else ""}
                    rows.append(row)
                    print(row, flush=True)
    with open(out / "ic_sensitivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
