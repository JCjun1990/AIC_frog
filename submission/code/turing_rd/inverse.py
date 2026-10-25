"""Building blocks of the (F, k) inversion of a steady pattern: steady states from random initial
states (batched, shared LU), polishing, and the feature distance used as the objective.

Steady states. Every column (F_i, k_i, seed_i) starts from the baseline protocol's initial state
(suggested IC with z scaled by the half-height, (1/2, 1/4) seeds, 5 % smooth noise; the seed fixes
it, so the same seeds at every candidate are common random numbers). SBDF2 with dt = 1 resolves
the spot-forming transient up to t_fast, then dt = 8 follows the slow spot replication and drift
(the scheme's fixed points are exact steady states for any dt). A column is handed to PTC / Newton
as soon as rms dv/dt < rate_tol and retired once the relative residual is below polish_tol; a
failed polish is retried after another `retry_after` time units. At t_max a column with
rms dv/dt < STEADY_RATE gets a last polish attempt. A column still changing faster than
give_up_rate after give_up_after is retired as "dynamic" (persistent spot replication / drift, two
orders of magnitude above STEADY_RATE). Columns still moving at t_max keep their last state.

Historical steadiness (`is_steady`, legacy-rate-v1): polished to polish_tol, or rms dv/dt < STEADY_RATE = 1e-5 at t_max (with
dt = 1 a change of 1e-5 per step, stricter than the organisers' "change between the last two steps
< 1e-4"). The target is a steady state, so a candidate's seed whose pattern is not steady gets the
same penalty as a wrong pattern type.

New reference workflows use full-pde-v1: the complete, unprojected, dual-equation residual
measured on the returned float64 fields must pass the fixed acceptance policy. Neither a
small dv/dt nor a reported polish success substitutes for this measurement.

Objective. Features are compared after the same measurement operator is applied to target and
simulation: heat-kernel low-pass at the target's cut-off, then projection on the physical range
(both are no-ops up to O(filter) for a smooth steady state). Per scored criterion (wavelength,
orientation, defects) the RMS of the feature errors in units of their tolerances is formed; the
distance is the RMS over the criteria, so each criterion weighs the same. A pattern of another
type than the target, or one that is not steady, gets TYPE_PENALTY.
"""
from __future__ import annotations

import time

import numpy as np

from .batch import BatchIntegrator
from .geometry import Geometry
from .initial import perturbed_ic
from .solver import GSParams
from .steady import ptc_polish
from .quality import full_residual_record, strict_full_steady

# criterion -> feature -> (error kind, tolerance). Tolerances are of the order of the seed-to-seed
# spread of a polished spot pattern at one parameter set (reported by scripts/invert_target.py).
OBJECTIVE = {
    "wavelength": {"wavelength": ("log", 0.05), "kappa_v": ("log", 0.05)},
    "orientation": {"coherence": ("diff", 0.05), "axial_order": ("diff", 0.10)},
    "defects": {"spot_defect_frac": ("diff", 0.15), "endpoint_density": ("diff", 0.20),
                "junction_density": ("diff", 0.10), "skeleton_density": ("diff", 0.20)},
}
TYPE_PENALTY = 5.0
STEADY_RATE = 1e-5
LEGACY_REFERENCE_PROTOCOL = "legacy-rate-v1"
STRICT_REFERENCE_PROTOCOL = "full-pde-v1"


def is_steady(rec: dict) -> bool:
    """Historical legacy-rate-v1 only. New reference workflows use reference_usable()."""
    return bool(rec["polished"] or (rec["status"] == "moving" and rec["rate"] < STEADY_RATE))


def reference_quality(rec):
    """Do not infer a full residual from a rate, polish flag, or spectral summary."""
    from .quality import finite
    rate = rec.get("rate")
    full = strict_full_steady(rec)
    from .spectral_policy import INVERSION_SCREEN_CONFIGS, evaluate_spectra
    sample = rec.get("spectral_report")
    diagnostic = rec.get("full_steady_diagnostic")
    decision = evaluate_spectra([sample] if isinstance(sample, dict) else [],
        diagnostic.get("value") if isinstance(diagnostic, dict) else None, configs=INVERSION_SCREEN_CONFIGS,
        scope="strict_reference_single_shift_finite_check")
    return {"reference_protocol": STRICT_REFERENCE_PROTOCOL,
            "low_rate": bool(finite(rate) and 0 <= rate < STEADY_RATE),
            "legacy_polished_flag": rec.get("polished") is True,
            "newton_polished": bool(rec.get("polished") is True and rec.get("status") != "uniform"),
            "polish_scope": "reported PTC/Newton success, not proof of full stationarity or Newton iteration history",
            "full_steady": full, "spectrally_screened": bool(full and decision["numerical_check_passed"])}


def reference_usable(rec, protocol=STRICT_REFERENCE_PROTOCOL):
    if protocol == STRICT_REFERENCE_PROTOCOL:
        return strict_full_steady(rec)
    if protocol == LEGACY_REFERENCE_PROTOCOL:
        return is_steady(rec)
    raise ValueError(f"unknown inversion reference protocol: {protocol}")


def feature_errors(sim: dict, tgt: dict, objective=OBJECTIVE) -> dict:
    """Per-feature error in tolerance units (signed: simulation minus target)."""
    out = {}
    for grp in objective.values():
        for name, (kind, tol) in grp.items():
            a, b = sim.get(name, np.nan), tgt.get(name, np.nan)
            if kind == "log":
                out[name] = np.log(max(a, 1e-12) / max(b, 1e-12)) / tol
            else:
                out[name] = (a - b) / tol
    return out


def feature_distance(sim: dict, tgt: dict, objective=OBJECTIVE) -> tuple[float, dict]:
    """Distance (RMS over criteria of the per-criterion RMS error) and the per-criterion values."""
    if sim.get("type") != tgt.get("type"):
        return TYPE_PENALTY, {g: TYPE_PENALTY for g in objective}
    z = feature_errors(sim, tgt, objective)
    per = {g: float(np.sqrt(np.nanmean([z[n] ** 2 for n in grp]))) for g, grp in objective.items()}
    return float(np.sqrt(np.mean([v ** 2 for v in per.values()]))), per


def initial_states(geom: Geometry, seeds):
    ics = [perturbed_ic(geom, seed=int(s), mode="seeded", normalize_z=True) for s in seeds]
    return np.stack([a for a, _ in ics], 1), np.stack([b for _, b in ics], 1)


def steady_states(geom: Geometry, F, k, seeds, Du: float, Dv: float, t_max: float = 40000.0,
                  t_fast: float = 500.0, dt: float = 8.0, chunk: float = 2000.0, rate_tol: float = 1e-6,
                  polish_tol: float = 1e-10, retry_after: float = 8000.0, polish: bool = True,
                  max_iter: int = 25, give_up_rate: float = 3e-4, give_up_after: float = 20000.0,
                  log=None) -> list[dict]:
    """Integrate columns (F[i], k[i], seeds[i]) to steady states; one dict per column, input order."""
    F, k, seeds = np.broadcast_arrays(np.asarray(F, float), np.asarray(k, float), np.asarray(seeds))
    B = len(seeds)
    U0, V0 = initial_states(geom, seeds)
    t0 = time.time()
    fast = BatchIntegrator(geom, Du, Dv, dt=1.0)
    fast.start(U0, V0, F, k)
    fast.advance(int(t_fast) - 1)
    bi = BatchIntegrator(geom, Du, Dv, dt=dt)
    bi.start(fast.u, fast.v, F, k)
    del fast
    out: list[dict | None] = [None] * B
    next_try = np.full(B, -np.inf)

    def finish(rate, j, u, v, polished, hist=None, kind="steady"):
        res = hist[-1]["rel_residual"] if hist else (0.0 if kind == "uniform" else np.nan)
        out[j] = {"u": np.array(u), "v": np.array(v), "t": t_fast + bi.t, "rate": float(rate), "polished": polished,
                  "residual": res, "ptc_iters": len(hist) - 1 if hist else 0, "status": kind,
                  "F": float(F[j]), "k": float(k[j]), "seed": int(seeds[j])}
        out[j]["full_steady_diagnostic"] = full_residual_record(geom, GSParams(float(F[j]), float(k[j]), Du, Dv), u, v)
        out[j]["reference_quality"] = reference_quality(out[j])

    while bi.size:
        n = int(np.ceil(min(chunk, t_max - t_fast - bi.t) / dt - 1e-9))
        if n > 0:
            bi.advance(n)
        t = t_fast + bi.t
        rates = bi.rates()
        keep = np.ones(bi.size, bool)
        for jc, j in enumerate(bi.ids):
            u, v = bi.u[:, jc], bi.v[:, jc]
            if not np.all(np.isfinite(v)):
                finish(rates[jc], j, u, v, False, kind="blow-up")
                keep[jc] = False
            elif v.max() < 1e-3:                       # decayed to the trivial state (1, 0)
                finish(rates[jc], j, np.ones_like(u), np.zeros_like(v), True, kind="uniform")
                keep[jc] = False
            elif polish and t >= give_up_after and rates[jc] > give_up_rate:
                finish(rates[jc], j, u, v, False, kind="dynamic")
                keep[jc] = False
            elif polish and t >= next_try[j] and (rates[jc] < rate_tol or
                                                  (t >= t_max - 0.5 * dt and rates[jc] < STEADY_RATE)):
                p = GSParams(float(F[j]), float(k[j]), Du, Dv)
                up, vp, hist = ptc_polish(geom, p, u, v, tol=polish_tol, dtau0=1e3, max_iter=max_iter)
                if hist[-1]["rel_residual"] < polish_tol:
                    finish(rates[jc], j, up, vp, True, hist)
                    keep[jc] = False
                else:
                    next_try[j] = t + retry_after
        if log:
            log(f"    t={t:.0f} running={int(keep.sum())} wall={time.time() - t0:.0f}s")
        if t >= t_max - 0.5 * dt:
            for jc, j in enumerate(bi.ids):
                if keep[jc]:
                    finish(rates[jc], j, bi.u[:, jc], bi.v[:, jc], False, kind="moving")
            break
        if not keep.all():
            bi.keep(keep)
    for o in out:
        o["wall_s"] = time.time() - t0
    return out
