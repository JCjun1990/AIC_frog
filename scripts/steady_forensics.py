"""Discretisation forensics under the steady-state hypothesis for Solution.mat.

If Solution.mat is the converged steady state of the organisers' discrete Gray-Scott system, then
with *their* discretisation the steady equations

    Du (-K u) + F M(1 - u) - N(u v^2) = 0,      Dv (-K v) - (F + k) M v + N(u v^2) = 0

hold on the raw data up to their solver tolerance (K stiffness, M mass, N the discrete nonlinear
term), so a least-squares fit of (Du, F) and (Dv, F + k) leaves R^2 -> 1 and lands on simple numbers.
This script builds the discretisations a MATLAB user would plausibly write for a quad mesh and
reports, for each, the fitted parameters and R^2 (rows weighted per unit area).

Families: P1 triangles (three ways of splitting the quads) with lumped, Voronoi or consistent mass
and nodal ("group") or exactly integrated nonlinear term; bilinear Q1 elements on the quads;
graph Laplacians. Also reports the steady integral identities F = int uv^2 / int(1-u) and
F + k = int uv^2 / int v, which hold for any mass-conserving discretisation.

Writes results/steady_forensics/summary.md and fits.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd.geometry import cotan_laplacian, robust_laplacian  # noqa: E402
from turing_rd.mesh import load_frog  # noqa: E402

OUT = ROOT / "results" / "steady_forensics"
DU0, DV0 = 2.78e-4, 1.39e-4

# 6-point Dunavant rule, exact for degree 4 on triangles (barycentric coordinates, weights sum to 1)
_A, _B = 0.816847572980459, 0.091576213509771
_C, _D = 0.108103018168070, 0.445948490915965
TRI_Q = np.array([[_A, _B, _B], [_B, _A, _B], [_B, _B, _A], [_C, _D, _D], [_D, _C, _D], [_D, _D, _C]])
TRI_W = np.array([0.109951743655322] * 3 + [0.223381589678011] * 3)


def tri_areas(V, T):
    return 0.5 * np.linalg.norm(np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)


def assemble(T, local, n):
    """Sum per-element (m, a, a) local matrices into an n x n sparse matrix."""
    a = T.shape[1]
    rows = np.repeat(T, a, axis=1).ravel()
    cols = np.tile(T, (1, a)).ravel()
    return sp.coo_matrix((local.reshape(-1), (rows, cols)), shape=(n, n)).tocsr()


def p1_consistent_mass(V, T):
    A = tri_areas(V, T)
    loc = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    return assemble(T, A[:, None, None] * loc[None], len(V))


def voronoi_mass(V, T):
    """Mixed Voronoi areas (Meyer et al. 2003)."""
    n = len(V)
    M = np.zeros(n)
    P = [V[T[:, i]] for i in range(3)]
    A = tri_areas(V, T)
    ang = []
    for i in range(3):
        a, b = P[(i + 1) % 3] - P[i], P[(i + 2) % 3] - P[i]
        ang.append(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b)))
    ang = np.array(ang)                                  # (3, m) angle at vertex i
    obtuse = (ang > np.pi / 2).any(axis=0)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        eij2 = ((P[j] - P[i]) ** 2).sum(1)
        eik2 = ((P[k] - P[i]) ** 2).sum(1)
        vor = (eij2 / np.tan(ang[k]) + eik2 / np.tan(ang[j])) / 8.0
        area_i = np.where(obtuse, np.where(ang[i] > np.pi / 2, A / 2, A / 4), vor)
        np.add.at(M, T[:, i], area_i)
    return M


def p1_exact_nonlinear(V, T, u, v):
    """N_i = int phi_i u_h v_h^2 for P1 interpolants (exact quadrature)."""
    A = tri_areas(V, T)
    uq, vq = u[T] @ TRI_Q.T, v[T] @ TRI_Q.T              # (m, q)
    f = A[:, None] * TRI_W[None] * uq * vq ** 2          # (m, q)
    N = np.zeros(len(V))
    for i in range(3):
        np.add.at(N, T[:, i], (f * TRI_Q[None, :, i]).sum(1))
    return N


def q1_matrices(V, Q, order=2):
    """Bilinear isoparametric surface elements on quads: stiffness K, consistent mass M, plus quadrature data."""
    g = {2: (np.array([-1, 1]) / np.sqrt(3), np.array([1.0, 1.0])),
         3: (np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)]), np.array([5, 8, 5]) / 9)}[order]
    xi, wi = np.meshgrid(g[0], g[0], indexing="ij")
    w2 = np.outer(g[1], g[1]).ravel()
    xi, eta = xi.ravel(), wi.ravel()
    sx, sy = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
    N = 0.25 * (1 + sx[None] * xi[:, None]) * (1 + sy[None] * eta[:, None])          # (q, 4)
    dNx = 0.25 * sx[None] * (1 + sy[None] * eta[:, None])
    dNy = 0.25 * sy[None] * (1 + sx[None] * xi[:, None])
    X = V[Q]                                                                          # (m, 4, 3)
    t1 = np.einsum("qa,mak->mqk", dNx, X)
    t2 = np.einsum("qa,mak->mqk", dNy, X)
    g11, g12, g22 = (t1 * t1).sum(-1), (t1 * t2).sum(-1), (t2 * t2).sum(-1)
    det = g11 * g22 - g12 ** 2
    J = np.sqrt(det)                                                                  # (m, q)
    i11, i12, i22 = g22 / det, -g12 / det, g11 / det
    grad = np.stack([dNx, dNy], -1)                                                   # (q, 4, 2)
    Gi = np.stack([np.stack([i11, i12], -1), np.stack([i12, i22], -1)], -2)           # (m, q, 2, 2)
    Kloc = np.einsum("qad,mqde,qbe,mq,q->mab", grad, Gi, grad, J, w2)
    Mloc = np.einsum("qa,qb,mq,q->mab", N, N, J, w2)
    n = len(V)
    return assemble(Q, Kloc, n), assemble(Q, Mloc, n), (N, J, w2)


def q1_exact_nonlinear(Q, u, v, qdata):
    N, J, w2 = qdata
    uq, vq = u[Q] @ N.T, v[Q] @ N.T                       # (m, q)
    f = uq * vq ** 2 * J * w2[None]
    out = np.zeros(int(Q.max()) + 1)
    for a in range(4):
        np.add.at(out, Q[:, a], (f * N[None, :, a]).sum(1))
    return out


def graph_laplacian(Q, n, weights=None):
    E = np.sort(np.vstack([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]]), axis=1)
    E = np.unique(E, axis=0)
    w = np.ones(len(E)) if weights is None else weights(E)
    W = sp.coo_matrix((w, (E[:, 0], E[:, 1])), shape=(n, n))
    W = (W + W.T).tocsr()
    return sp.diags(np.asarray(W.sum(1)).ravel()) - W      # stiffness-like (positive semidefinite)


def fit(K, Mop, Nvec, u, v, area):
    """Least squares of the two steady equations; rows weighted by 1/sqrt(local area)."""
    w = 1 / np.sqrt(area)
    Mu = (lambda x: Mop @ x) if sp.issparse(Mop) else (lambda x: Mop * x)
    X1 = np.c_[-(K @ u), Mu(1 - u)] * w[:, None]
    y1 = Nvec * w
    X2 = np.c_[-(K @ v), -Mu(v)] * w[:, None]
    y2 = -Nvec * w
    p1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    p2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    r1, r2 = y1 - X1 @ p1, y2 - X2 @ p2
    R2u = 1 - r1 @ r1 / ((y1 - y1.mean()) @ (y1 - y1.mean()))
    R2v = 1 - r2 @ r2 / ((y2 - y2.mean()) @ (y2 - y2.mean()))
    # residual with the organisers' stated spot parameters (meaningful for metric operators)
    g1 = y1 - X1 @ np.array([DU0, 0.028])
    g2 = y2 - X2 @ np.array([DV0, 0.090])
    rel_given = np.sqrt((g1 @ g1 + g2 @ g2) / (y1 @ y1 + y2 @ y2))
    return {"Du": p1[0], "F": p1[1], "Dv": p2[0], "k": p2[1] - p1[1], "Dv/Du": p2[0] / p1[0],
            "R2u": R2u, "R2v": R2v, "rel_resid_given": rel_given}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mesh = load_frog(ROOT / "frog.mat")
    V, Q = mesh.V, mesh.quads
    n = len(V)
    S = sio.loadmat(str(ROOT / "Solution.mat"))["Solution"]
    u, v = S["u"][0, 0].ravel(), S["v"][0, 0].ravel()
    # orient quads outward (signed volume of the fan triangulation)
    T0 = np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    if np.einsum("ij,ij->i", V[T0[:, 0]], np.cross(V[T0[:, 1]], V[T0[:, 2]])).sum() < 0:
        Q = Q[:, ::-1]
    splits = {"diag 1-3": np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]),
              "diag 2-4": np.vstack([Q[:, [0, 1, 3]], Q[:, [1, 2, 3]]]),
              "short diag": None}
    d02 = np.linalg.norm(V[Q[:, 0]] - V[Q[:, 2]], axis=1)
    d13 = np.linalg.norm(V[Q[:, 1]] - V[Q[:, 3]], axis=1)
    use02 = d02 <= d13
    splits["short diag"] = np.vstack([np.where(use02[:, None], Q[:, [0, 1, 2]], Q[:, [0, 1, 3]]),
                                      np.where(use02[:, None], Q[:, [0, 2, 3]], Q[:, [1, 2, 3]])])
    rows = []
    for sname, T in splits.items():
        L, Mlump = cotan_laplacian(V, T)
        K = -L
        Mc = p1_consistent_mass(V, T)
        Mv = voronoi_mass(V, T)
        Nexact = p1_exact_nonlinear(V, T, u, v)
        area = Mlump
        r = u * v * v
        for mname, Mop, Nvec in [("lumped", Mlump, Mlump * r), ("Voronoi", Mv, Mv * r),
                                 ("consistent, nodal nonlinear", Mc, Mc @ r),
                                 ("consistent, exact nonlinear", Mc, Nexact),
                                 ("lumped, exact nonlinear", Mlump, Nexact)]:
            rows.append({"family": "P1 cotan", "variant": f"{sname} / {mname}",
                         **fit(K, Mop, Nvec, u, v, area)})
    Lr, Mr = robust_laplacian(V, splits["short diag"])
    rows.append({"family": "P1 robust", "variant": "intrinsic Delaunay / lumped",
                 **fit(-Lr, Mr, Mr * u * v * v, u, v, Mr)})
    for order in (2, 3):
        Kq, Mq, qd = q1_matrices(V, Q, order)
        Ml = np.asarray(Mq.sum(1)).ravel()
        Nq = q1_exact_nonlinear(Q, u, v, q1_matrices(V, Q, 3)[2])
        r = u * v * v
        for mname, Mop, Nvec in [("lumped", Ml, Ml * r), ("consistent, nodal nonlinear", Mq, Mq @ r),
                                 ("consistent, exact nonlinear", Mq, Nq)]:
            rows.append({"family": f"Q1 bilinear ({order}x{order} Gauss)", "variant": mname,
                         **fit(Kq, Mop, Nvec, u, v, Ml)})
    _, Mbar = cotan_laplacian(V, splits["short diag"])
    hbar2 = np.mean(((V[Q[:, 0]] - V[Q[:, 1]]) ** 2).sum(1))
    for wname, wf, Mop in [
        ("unweighted, unit mass", None, np.ones(n)),
        ("unweighted, vertex area", None, Mbar),
        ("1/|e|^2 weights, unit mass", lambda E: 1 / ((V[E[:, 0]] - V[E[:, 1]]) ** 2).sum(1), np.ones(n)),
        ("1/|e| weights, vertex area", lambda E: 1 / np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1), Mbar)]:
        Kg = graph_laplacian(Q, n, wf)
        r = u * v * v
        res = fit(Kg, Mop, Mop * r, u, v, Mop if np.ndim(Mop) else np.ones(n))
        if wname.startswith("unweighted, unit"):
            res["Du"] *= hbar2; res["Dv"] *= hbar2          # report as a physical coefficient
        rows.append({"family": "graph", "variant": wname, **res})

    rows.sort(key=lambda r: -(r["R2u"] + r["R2v"]))
    keys = list(rows[0])
    with open(OUT / "fits.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    lines = ["# 离散格式取证（稳态假设）\n",
             "每种离散在原始数据上拟合两条稳态方程；若正好是组委会的离散且为收敛稳态，R² 应趋近 1。"
             "「题给参数残差」= 用 Du = 2.78e-4、Dv = 1.39e-4、F = 0.028、k = 0.062 时的相对残差。\n",
             "| 离散族 | 变体 | R²(u) | R²(v) | Du | F | Dv | k | Dv/Du | 题给参数残差 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['variant']} | {r['R2u']:.4f} | {r['R2v']:.4f} | {r['Du']:.3g} | {r['F']:.4f} | "
                     f"{r['Dv']:.3g} | {r['k']:.4f} | {r['Dv/Du']:.3f} | {r['rel_resid_given']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
