"""Solver verification.

1. Temporal order of SBDF1 / SBDF2 against a fine-step reference (expect slopes 1 and 2).
2. Backend agreement: SciPy LU vs PyTorch CG (float64 and float32).
3. Batched CG (different F, k, Du per column) vs individual LU runs.
4. Gradients: torch.autograd.gradcheck w.r.t. F, k, Du, Dv and the initial state, both backends;
   gradient checkpointing reproduces the plain gradient.
5. Optional throughput benchmark on a device (e.g. --bench-device cuda on the RTX 4080).

Writes results/verification/solver.json and time_order.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, GrayScottSolver, GSParams, STRIPES, SPOTS, icosphere, load_frog  # noqa: E402
from turing_rd.initial import perturbed_ic  # noqa: E402


def mnorm(geom, a):
    return float(np.sqrt(geom.M @ (a * a) / geom.area))


def time_order(report):
    g = Geometry(icosphere(4), cache_dir=None)
    u0, v0 = perturbed_ic(g, eps=0.05, seed=0)
    T = 40.0
    ref = GrayScottSolver(g, dt=1 / 64, scheme="sbdf2")
    with torch.no_grad():
        ur, vr = (x.numpy() for x in ref.rollout(u0, v0, STRIPES, int(T * 64)))
    dts = [4.0, 2.0, 1.0, 0.5, 0.25]
    res = {}
    for scheme in ("sbdf1", "sbdf2"):
        errs = []
        for dt in dts:
            s = GrayScottSolver(g, dt=dt, scheme=scheme)
            with torch.no_grad():
                u, v = (x.numpy() for x in s.rollout(u0, v0, STRIPES, int(round(T / dt))))
            errs.append(mnorm(g, u - ur) + mnorm(g, v - vr))
        slope = float(np.polyfit(np.log(dts[1:]), np.log(errs[1:]), 1)[0])
        res[scheme] = {"dt": dts, "error": errs, "order": slope}
        print(f"time order {scheme}: errors {np.array(errs)} -> slope {slope:.2f}", flush=True)
    report["time_order"] = res
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for scheme, r in res.items():
        ax.loglog(r["dt"], r["error"], "o-", label=f"{scheme.upper()}, slope {r['order']:.2f}")
    ax.set_xlabel("dt")
    ax.set_ylabel("error at t = 40 (rms u + rms v)")
    ax.set_title("Temporal convergence, sphere L4, stripes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "verification" / "time_order.png", dpi=120)


def backends(report):
    g = Geometry(icosphere(4), cache_dir=None)
    u0, v0 = perturbed_ic(g, eps=0.05, seed=0)
    out = {}
    with torch.no_grad():
        u_lu, v_lu = GrayScottSolver(g, backend="lu").rollout(u0, v0, SPOTS, 300)
        for dtype in (torch.float64, torch.float32):
            s = GrayScottSolver(g, backend="cg", dtype=dtype, cg_rtol=1e-12 if dtype == torch.float64 else 1e-6)
            u, v = s.rollout(u0, v0, SPOTS, 300)
            out[str(dtype)] = float(max((u.double() - u_lu).abs().max(), (v.double() - v_lu).abs().max()))
        # batch: three parameter sets in one CG solve vs three LU runs
        sets = [GSParams(0.028, 0.062), GSParams(0.054, 0.063), GSParams(0.040, 0.060, 3e-4, 1e-4)]
        Pb = {k: torch.tensor([getattr(p, k) for p in sets], dtype=torch.float64) for k in ("F", "k", "Du", "Dv")}
        U0 = np.repeat(u0[:, None], 3, 1)
        V0 = np.repeat(v0[:, None], 3, 1)
        ub, vb = GrayScottSolver(g, backend="cg", cg_rtol=1e-12).rollout(U0, V0, Pb, 300)
        diffs = []
        for j, p in enumerate(sets):
            u1, v1 = GrayScottSolver(g, backend="lu").rollout(u0, v0, p, 300)
            diffs.append(float(max((ub[:, j] - u1).abs().max(), (vb[:, j] - v1).abs().max())))
        out["batched_cg_vs_lu"] = diffs
    report["backends_max_abs_diff_after_300_steps"] = out
    print("backend agreement:", out, flush=True)


def gradients(report):
    g = Geometry(icosphere(1), cache_dir=None)
    u0, v0 = perturbed_ic(g, eps=0.05, seed=0)
    w = torch.as_tensor(np.random.default_rng(1).standard_normal(g.n))
    out = {}
    for backend in ("lu", "cg"):
        s = GrayScottSolver(g, dt=1.0, scheme="sbdf2", backend=backend, cg_rtol=1e-13)

        def loss(F, k, Du, Dv, u_init):
            P = {"F": F, "k": k, "Du": Du, "Dv": Dv}
            u, v = s.rollout(u_init, torch.as_tensor(v0), P, 6)
            return (s.op.M * (v * v + w * u)).sum()

        args = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (0.04, 0.06, 0.02, 0.01)]
        args.append(torch.as_tensor(u0).clone().requires_grad_(True))
        ok = torch.autograd.gradcheck(loss, args, eps=1e-6, atol=1e-8, rtol=1e-5)
        out[backend] = bool(ok)
        print(f"gradcheck {backend}: {ok}", flush=True)
    # checkpointing gives the same gradient
    s = GrayScottSolver(g, dt=1.0, scheme="sbdf2")
    grads = []
    for ck in (0, 5):
        F = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
        Du = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
        u, v = s.rollout(u0, v0, {"F": F, "k": torch.tensor(0.06, dtype=torch.float64), "Du": Du,
                                  "Dv": torch.tensor(0.01, dtype=torch.float64)}, 40, checkpoint_every=ck)
        (s.op.M * v * v).sum().backward()
        grads.append(np.array([F.grad.item(), Du.grad.item()]))
    out["checkpoint_rel_diff"] = float(np.abs(grads[0] - grads[1]).max() / np.abs(grads[0]).max())
    print("checkpoint vs plain gradient, rel diff:", out["checkpoint_rel_diff"], flush=True)
    report["gradcheck"] = out


def bench(report, device, batches):
    g = Geometry(load_frog(ROOT / "frog.mat"), cache_dir=None)
    u0, v0 = perturbed_ic(g, eps=0.05, seed=0)
    res = {"torch": torch.__version__, "device": device}
    if device.startswith("cuda"):
        res.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    steps = 200
    with torch.no_grad():
        s = GrayScottSolver(g, backend="lu")
        t = time.time()
        s.rollout(u0, v0, SPOTS, steps)
        res["lu_cpu_float64_ms_per_step"] = 1e3 * (time.time() - t) / steps
        for batch in batches:
            s = GrayScottSolver(g, backend="cg", device=device, dtype=torch.float32, cg_rtol=1e-6)
            F = torch.linspace(0.02, 0.06, batch)
            P = {"F": F, "k": torch.full((batch,), 0.062), "Du": torch.full((batch,), 2.78e-4),
                 "Dv": torch.full((batch,), 1.39e-4)}
            U0, V0 = np.repeat(u0[:, None], batch, 1), np.repeat(v0[:, None], batch, 1)
            s.rollout(U0, V0, P, 5)                                      # warm-up
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.time()
            u, v = s.rollout(U0, V0, P, steps)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = (time.time() - t) / steps
            res[f"cg_float32_batch{batch}_ms_per_step"] = 1e3 * dt
            res[f"cg_float32_batch{batch}_ms_per_simulation_step"] = 1e3 * dt / batch
            print(f"  batch {batch:5d}: {1e3 * dt:8.2f} ms/step = {1e3 * dt / batch:.4f} ms per simulation-step",
                  flush=True)
        # accuracy against float64 LU: the last float32 batch (first, middle and last column), and a
        # float64 CG run on the same device, which separates float32 rounding from device-path errors
        def maxdiff(a, b):
            return float((a.double().cpu() - b).abs().max())

        refs = {}
        for j in sorted({0, batch // 2, batch - 1}):
            pj = GSParams(float(F[j]), 0.062, 2.78e-4, 1.39e-4)
            refs[j] = (pj, *GrayScottSolver(g, backend="lu").rollout(u0, v0, pj, steps))
        res["max_abs_diff_float32_vs_lu"] = max(max(maxdiff(u[:, j], ur), maxdiff(v[:, j], vr))
                                                for j, (_, ur, vr) in refs.items())
        res["max_v_reference"] = max(float(vr.abs().max()) for _, _, vr in refs.values())
        pj, ur, vr = refs[batch // 2]
        s = GrayScottSolver(g, backend="cg", device=device, dtype=torch.float64, cg_rtol=1e-12)
        u64, v64 = s.rollout(u0, v0, pj, steps)
        res["max_abs_diff_float64_cg_vs_lu"] = max(maxdiff(u64, ur), maxdiff(v64, vr))
    report["benchmark_frog"] = res
    print("benchmark:", json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-device", default=None, help="e.g. cuda, or cpu")
    ap.add_argument("--bench-batch", type=int, nargs="+", default=[64], help="one or more batch sizes")
    ap.add_argument("--skip", nargs="*", default=[])
    args = ap.parse_args()
    (ROOT / "results" / "verification").mkdir(parents=True, exist_ok=True)
    path = ROOT / "results" / "verification" / "solver.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    if "order" not in args.skip:
        time_order(report)
    if "backends" not in args.skip:
        backends(report)
    if "grad" not in args.skip:
        gradients(report)
    if args.bench_device:
        bench(report, args.bench_device, args.bench_batch)
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
