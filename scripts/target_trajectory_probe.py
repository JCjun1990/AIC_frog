"""Is Solution.mat on the trajectory of (given F, k, Du, Dv + the suggested IC)?

Integrates the deterministic suggested IC on the frog with a given Laplacian and parameter set and
tracks the area-weighted pointwise correlation between the simulated and target fields.

    python scripts/target_trajectory_probe.py robust spots     (laplacian: robust | cotan_naive)
"""
import sys, json, numpy as np, scipy.io as sio, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd import Geometry, GrayScottSolver, SPOTS, STRIPES
from turing_rd.mesh import Mesh, load_frog
from turing_rd.initial import suggested_ic
from turing_rd.diagnostics import pattern_stats
lap, pname = sys.argv[1], sys.argv[2]
m = load_frog(ROOT / "frog.mat")
if lap == "cotan_naive":                       # MATLAB/gptoolbox-style split [1 2 3; 1 3 4]
    Q = m.quads; m = Mesh(m.V, np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]), "frog", Q); lap_kind = "cotan"
else:
    lap_kind = "robust"
g = Geometry(m, laplacian=lap_kind, cache_dir=None)
S = sio.loadmat(ROOT / "Solution.mat")["Solution"]
ut, vt = S["u"][0, 0].ravel(), S["v"][0, 0].ravel()
def wcorr(a, b):
    a = a - g.mean(a); b = b - g.mean(b)
    return float((g.M @ (a * b)) / np.sqrt((g.M @ (a * a)) * (g.M @ (b * b)) + 1e-300))
p = SPOTS if pname == "spots" else STRIPES
u0, v0 = suggested_ic(g)
best = {"corr_v": -2}; hist = []
def cb(step, t, u, v, up, vp):
    u, v = u.numpy(), v.numpy()
    cv, cu = wcorr(v, vt), wcorr(u, ut)
    rel = g.rms(v - vt) / g.rms(vt)
    hist.append((t, cv, cu, rel))
    if cv > best["corr_v"]:
        best.update(t=t, corr_v=cv, corr_u=cu, rel_l2_v=rel, vmax=float(v.max()), n=pattern_stats(g, v).get("n_components"))
    return False
s = GrayScottSolver(g, dt=1.0)
with torch.no_grad():
    s.rollout(u0, v0, p, 20000, record_every=10, callback=cb)
h = np.array(hist)
for t in (10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000):
    i = np.argmin(np.abs(h[:, 0] - t)); print(f"  {lap}/{pname} t={h[i,0]:6.0f} corr_v={h[i,1]:+.3f} corr_u={h[i,2]:+.3f} relL2_v={h[i,3]:.2f}")
print("BEST", json.dumps({"lap": lap, "params": pname, **best}))
