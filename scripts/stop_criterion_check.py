"""Check the organisers' steady criterion: "the last two time levels differ by less than 1e-4".

    python scripts/stop_criterion_check.py target      rates of change of Solution.mat itself
    python scripts/stop_criterion_check.py run NAME    integrate an IC variant until the criterion holds

target: |du/dt|, |dv/dt| of the target under the given spot parameters (raw, and clipped to the
        physical range u in [0, 1], v >= 0). With an explicit step the per-step change is dt * rate,
        so the criterion only holds for dt <= 1e-4 / max rate; also one implicit SBDF1 step per dt.
run:    IC variants (rules = the suggested formula with raw z, as in the rules):
            rules, rules_noise0/1/2 (+ 5% smooth noise), protocol (our z-normalised seeded IC)
        spot parameters, SBDF2, dt = 1, until max |change per step| < 1e-4 (t >= 20) or t = 20000;
        pattern statistics there vs the target.
Writes results/stop_criterion/*.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, GrayScottSolver, SPOTS, load_frog  # noqa: E402
from turing_rd.diagnostics import pattern_stats  # noqa: E402
from turing_rd.initial import perturbed_ic, suggested_ic  # noqa: E402

OUT = ROOT / "results" / "stop_criterion"
OUT.mkdir(parents=True, exist_ok=True)
g = Geometry(load_frog(ROOT / "frog.mat"))
S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
ut, vt = S["u"][0, 0].ravel().astype(float), S["v"][0, 0].ravel().astype(float)
p = SPOTS


def wcorr(a, b):
    a, b = a - g.mean(a), b - g.mean(b)
    return float((g.M @ (a * b)) / np.sqrt((g.M @ (a * a)) * (g.M @ (b * b)) + 1e-300))


def stats(u, v):
    s = pattern_stats(g, v)
    return {"type": s.get("type"), "n_components": s.get("n_components"), "wavelength": s.get("wavelength"),
            "v_max": float(v.max()), "v_min": float(v.min()), "frac_v_neg": float((g.M @ (v < 0)) / g.area),
            "corr_v_with_target": wcorr(v, vt)}


def rates(u, v):
    fu = p.Du * g.lap(u) - u * v * v + p.F * (1 - u)
    fv = p.Dv * g.lap(v) + u * v * v - (p.F + p.k) * v
    return {"max_du_dt": float(np.abs(fu).max()), "max_dv_dt": float(np.abs(fv).max()),
            "rms_du_dt": g.rms(fu), "rms_dv_dt": g.rms(fv)}


if sys.argv[1] == "target":
    res = {"target": stats(ut, vt)}
    for name, (u, v) in {"raw": (ut, vt), "clipped": (np.clip(ut, 0, 1), np.clip(vt, 0, None))}.items():
        r = rates(u, v)
        r["dt_for_1e-4_explicit"] = 1e-4 / max(r["max_du_dt"], r["max_dv_dt"])
        steps = {}
        for dt in (1.0, 0.1, 0.01, 0.001):
            with torch.no_grad():
                u1, v1 = GrayScottSolver(g, dt=dt, scheme="sbdf1").rollout(u, v, p, 1)
            steps[str(dt)] = max(float(np.abs(u1.numpy() - u).max()), float(np.abs(v1.numpy() - v).max()))
        r["max_change_one_sbdf1_step"] = steps
        res[name] = r
    (OUT / "target.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    sys.exit()

name = sys.argv[2]
if name == "rules":
    u0, v0 = suggested_ic(g)
elif name.startswith("rules_noise"):
    u0, v0 = perturbed_ic(g, eps=0.05, seed=int(name[-1]), mode="additive", normalize_z=False)
elif name == "protocol":
    u0, v0 = perturbed_ic(g, eps=0.05, seed=0, mode="seeded", normalize_z=True)
else:
    raise SystemExit(f"unknown IC {name}")
log = {"ic": name, "t_stop": None}


def cb(step, t, u, v, up, vp):
    if up is None:
        return False
    change = max(float((u - up).abs().max()), float((v - vp).abs().max()))
    if step % 1000 == 0:
        log.setdefault("change_every_1000", []).append([t, change])
    if t >= 20 and change < 1e-4:
        log.update(t_stop=t, change_at_stop=change, **stats(u.numpy(), v.numpy()))
        return True
    return False


with torch.no_grad():
    u, v = GrayScottSolver(g, dt=1.0).rollout(u0, v0, p, 20000, record_every=1, callback=cb)
if log["t_stop"] is None:
    log.update(t_end=20000, **stats(u.numpy(), v.numpy()))
(OUT / f"{name}.json").write_text(json.dumps(log, indent=2))
print(name, {k: log.get(k) for k in ("t_stop", "type", "n_components", "wavelength", "v_max", "corr_v_with_target")})
