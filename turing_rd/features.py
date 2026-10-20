"""Pattern fingerprint of one snapshot (u, v): the features used for statistical inversion.

Base groups (FeatureExtractor.names, 37 features; the band dataset and refine_cmaes.py use these):
    morphology  wavelength, spot count, area fraction, compactness, Euler number per spot,
                perimeter per unit area
    spot sizes  10 / 50 / 90 % quantiles of spot areas (fraction of the surface), their spread
    amplitude   v quantiles (50, 90, 99 %, max), u quantiles (1, 10, 50 %), mean u and v inside and
                outside the spots
    spectrum    share of the variance of v in Laplace-Beltrami wavenumber bands kappa = sqrt(lambda),
                the variance beyond the resolved modes, and spectral centroids of u and v

Extended groups (FeatureExtractor(..., extended=True), appended after the base 37):
    orientation coherence     energy-weighted mean of the local structure-tensor coherence
                              (mu1 - mu2) / (mu1 + mu2) of grad v, window sigma = orient_sigma:
                              ~0.1-0.3 for spots (gradients point every way), larger for stripes
                axial_order   energy-weighted mean of coherence * cos(2 phi), phi = angle between the
                              dominant gradient and the body axis projected on the tangent plane:
                              > 0 bands across the body, < 0 stripes along it, 0 no preference
    defects     spot_irregular_frac     spots with isoperimetric ratio 4 pi A / P^2 < Q_IRREGULAR
                spot_size_outlier_frac  spots whose area is off the median by more than SIZE_FACTOR
                spot_nondisk_frac       spots whose Euler characteristic is not 1 (rings, holes)
                spot_defect_frac        spots with any of the three defects
                endpoint_density        free ends of the pruned skeleton of the high-v phase,
                junction_density        and its branch points (>= 3 branches), per lambda_ref^2
                skeleton_density        skeleton length * lambda_ref / area (~1 for parallel stripes)
    The spot measures address merged or broken spots, the skeleton measures stripe terminations and
    Y-junctions. A round spot thins to a point and an isolated skeleton piece shorter than
    prune_length is dropped, so a regular spot pattern has endpoint density ~0; a merged (dumbbell)
    spot leaves a segment with two free ends.

Body axis: principal axis of largest second moment of the surface (the frog's head-tail axis); if
the two largest moments are within 5 % (sphere), the z axis of the problem's cylindrical
coordinates.

Deliberately left out: min v and max u. The target has v < 0 and u > 1 at some vertices (solver
artefacts), which no positivity-preserving simulation can reproduce.
"""
from __future__ import annotations

import heapq

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components, dijkstra, minimum_spanning_tree

from .diagnostics import _contour_length, contour_segments
from .geometry import Geometry

KAPPA_EDGES = np.array([0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 33.0])
GROUPS = {
    "morphology": ["wavelength", "n_spots", "area_fraction", "compactness", "euler_per_spot", "perimeter_density"],
    "spot_sizes": ["size_q10", "size_q50", "size_q90", "size_spread"],
    "amplitude": ["v_q50", "v_q90", "v_q99", "v_max", "u_q01", "u_q10", "u_q50", "v_in", "u_in", "v_out", "u_out"],
    "spectrum": [f"P_k{int(a)}_{int(b)}" for a, b in zip(KAPPA_EDGES[:-1], KAPPA_EDGES[1:])]
                + ["P_unresolved", "kappa_v", "kappa_u"],
}
EXTENDED_GROUPS = {
    "orientation": ["coherence", "axial_order"],
    "defects": ["spot_irregular_frac", "spot_size_outlier_frac", "spot_nondisk_frac", "spot_defect_frac",
                "endpoint_density", "junction_density", "skeleton_density"],
}
Q_IRREGULAR = 0.75          # isoperimetric ratio below which a spot counts as irregular
SIZE_FACTOR = 1.75          # area ratio to the median beyond which a spot is a size outlier


def body_axis(geom: Geometry) -> np.ndarray:
    """Unit principal axis of the area-weighted second moment (z if the surface is near-isotropic)."""
    x = geom.V - (geom.M @ geom.V) / geom.area
    C = (x * geom.M[:, None]).T @ x / geom.area
    w, Q = np.linalg.eigh(C)
    if w[2] < 1.05 * w[1]:
        return np.array([0.0, 0.0, 1.0])
    a = Q[:, 2]
    return a * np.sign(a[np.argmax(np.abs(a))])


def _rings(F: np.ndarray, n: int):
    """Neighbours of every vertex in cyclic order around it (closed, consistently oriented mesh)."""
    src = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    frm = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    to = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    order = np.argsort(src, kind="stable")
    src, frm, to = src[order], frm[order], to[order]
    ptr = np.searchsorted(src, np.arange(n + 1))
    ring = np.empty_like(frm)
    for i in range(n):
        a, b = ptr[i], ptr[i + 1]
        nxt = dict(zip(frm[a:b].tolist(), to[a:b].tolist()))
        cur = int(frm[a])
        for j in range(a, b):
            ring[j] = cur
            cur = nxt[cur]
    return ptr, ring


class FeatureExtractor:
    def __init__(self, geom: Geometry, n_modes: int = 600, min_size: int = 3, extended: bool = False,
                 orient_sigma: float = 0.2, lambda_ref: float = 0.5, prune_length: float = 0.25):
        self.g, self.min_size, self.extended = geom, min_size, extended
        lam, phi = geom.eigenbasis(n_modes)
        self.kappa = np.sqrt(np.maximum(lam, 0))
        self.PhiTM = (phi * geom.M[:, None]).T                       # (K, n): M-weighted projection
        self.bins = np.clip(np.digitize(self.kappa, KAPPA_EDGES) - 1, 0, len(KAPPA_EDGES) - 2)
        groups = {**GROUPS, **EXTENDED_GROUPS} if extended else GROUPS
        self.names = [n for grp in groups.values() for n in grp]
        if extended:
            self._setup_extended(orient_sigma, lambda_ref, prune_length)

    # ---- extended-feature geometry, computed once per surface --------------------------------------
    def _setup_extended(self, orient_sigma, lambda_ref, prune_length):
        g = self.g
        V, F = g.V, g.F
        N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        twoA = np.linalg.norm(N, axis=1)
        nf = N / twoA[:, None]
        # P1 basis gradients: grad phi_i = n x (opposite edge) / 2A
        self.grad_basis = [np.cross(nf, V[F[:, (i + 2) % 3]] - V[F[:, (i + 1) % 3]]) / twoA[:, None] for i in range(3)]
        m = len(F)
        self.face_area = twoA / 2.0
        self.face_to_vertex = sp.csr_matrix((np.repeat(twoA / 6.0, 3), (F.T.ravel(), np.tile(np.arange(m), 3))),
                                            shape=(g.n, m))      # rows sum to the lumped mass M
        self.orient_sigma = orient_sigma
        tau = orient_sigma ** 2 / 2.0                                # heat kernel ~ Gaussian of std sigma
        self._smooth_steps = 4
        self._smooth_lu = spla.splu((sp.diags(g.M) - (tau / self._smooth_steps) * g.L).tocsc())
        nv = g.normals
        self.P_tan = np.eye(3)[None] - nv[:, :, None] * nv[:, None, :]
        self.axis = body_axis(g)
        at = self.P_tan @ self.axis
        norm = np.linalg.norm(at, axis=1)
        self.axis_ok = norm > 0.3                                   # away from the axis poles
        self.axis_t = at / np.maximum(norm, 1e-12)[:, None]
        self.ring_ptr, self.ring = _rings(F, g.n)
        el = np.linalg.norm(V[g.edges[:, 0]] - V[g.edges[:, 1]], axis=1)
        W = sp.coo_matrix((el, (g.edges[:, 0], g.edges[:, 1])), shape=(g.n, g.n))
        self.edge_len = (W + W.T).tocsr()
        self.lambda_ref, self.prune_length = lambda_ref, prune_length

    def _heat(self, X):
        for _ in range(self._smooth_steps):
            X = self._smooth_lu.solve(self.g.M[:, None] * X)
        return X

    def orientation(self, v) -> dict:
        """Local structure tensor of grad v, smoothed over orient_sigma, projected on the tangent plane."""
        g, F = self.g, self.g.F
        G = sum(v[F[:, i]][:, None] * self.grad_basis[i] for i in range(3))          # (m, 3) face gradients
        iu = np.triu_indices(3)
        Tf = (G[:, :, None] * G[:, None, :])[:, iu[0], iu[1]]                         # (m, 6)
        Tv = self._heat((self.face_to_vertex @ Tf) / g.M[:, None])
        T = np.empty((g.n, 3, 3))
        T[:, iu[0], iu[1]] = Tv
        T[:, iu[1], iu[0]] = Tv
        T = self.P_tan @ T @ self.P_tan
        w, Q = np.linalg.eigh(T)
        mu1, mu2 = np.maximum(w[:, 2], 0), np.maximum(w[:, 1], 0)
        c = (mu1 - mu2) / np.maximum(mu1 + mu2, 1e-300)
        e = (mu1 + mu2) * g.M                                                           # gradient energy weight
        cos2 = 2 * np.einsum("ij,ij->i", Q[:, :, 2], self.axis_t) ** 2 - 1
        ok = self.axis_ok
        coh = float(e @ c / max(e.sum(), 1e-300))
        axial = float((e * c * cos2)[ok].sum() / max(e[ok].sum(), 1e-300))
        coh_ok = float((e * c)[ok].sum() / max(e[ok].sum(), 1e-300))
        phi = 0.5 * np.degrees(np.arccos(np.clip(axial / max(coh_ok, 1e-12), -1, 1)))  # gradient vs axis
        return {"coherence": coh, "axial_order": axial, "stripe_angle_deg": 90.0 - phi}

    # ---- defects ---------------------------------------------------------------------------------
    def _runs(self, mask, verts):
        """Number of runs of `mask` vertices around each vertex in `verts` (branches leaving it)."""
        out = np.empty(len(verts), int)
        ptr, ring = self.ring_ptr, self.ring
        for j, p in enumerate(verts):
            m = mask[ring[ptr[p]:ptr[p + 1]]]
            out[j] = np.count_nonzero(m & ~np.roll(m, 1)) if not m.all() else 0
        return out

    def skeleton(self, hi, v) -> np.ndarray:
        """Ordered homotopic thinning of the vertex set `hi`: vertices are peeled in order of increasing v.

        A vertex is removed when its neighbours in the set form a single arc of its 1-ring (its link is
        contractible, so components and holes are kept) and it is not a free end (>= 2 such neighbours).
        Peeling along the level sets of v leaves the crest lines: a single-peaked spot shrinks to its
        peak, a stripe to its ridge, and two spots merged through a saddle to a segment between the peaks.
        """
        S = hi.copy()
        ptr, ring = self.ring_ptr, self.ring
        out_nb = np.add.reduceat((~S[ring]).astype(np.int64), ptr[:-1]) > 0
        heap = [(float(v[p]), int(p)) for p in np.flatnonzero(S & out_nb)]
        heapq.heapify(heap)
        while heap:
            _, p = heapq.heappop(heap)
            if not S[p]:
                continue
            nb = ring[ptr[p]:ptr[p + 1]]
            m = S[nb]
            cnt = np.count_nonzero(m)
            if cnt <= 1 or cnt == len(m) or np.count_nonzero(m & ~np.roll(m, 1)) != 1:
                continue                      # re-examined when one of its neighbours is removed
            S[p] = False
            for q in nb[m]:
                heapq.heappush(heap, (float(v[q]), int(q)))
        return S

    def skeleton_stats(self, hi, v) -> dict:
        """Free ends, branch points and length of the skeleton after dropping short pieces and spurs."""
        g, Lmin = self.g, self.prune_length
        K = self.skeleton(hi, v)
        for _ in range(2):                                   # drop short pieces, then spurs, then recount
            idx = np.flatnonzero(K)
            if len(idx) == 0:
                break
            W = self.edge_len[idx][:, idx]
            nc, lab = connected_components(W, directed=False)
            mst = minimum_spanning_tree(W).tocoo()
            comp_len = np.bincount(lab[mst.row], mst.data, minlength=nc)
            short = comp_len[lab] < Lmin
            K[idx[short]] = False
            idx = np.flatnonzero(K)
            if len(idx) == 0:
                break
            W = self.edge_len[idx][:, idx]
            r = self._runs(K, idx)
            ends, junc = np.flatnonzero(r == 1), np.flatnonzero(r >= 3)
            if len(ends) == 0 or len(junc) == 0:
                continue
            D, pred = dijkstra(W, directed=False, indices=ends, limit=Lmin, return_predecessors=True)
            is_junc = np.zeros(len(idx), bool)
            is_junc[junc] = True
            drop = []
            for a, e in enumerate(ends):
                reach = np.flatnonzero(np.isfinite(D[a]) & is_junc)
                if len(reach) == 0:
                    continue
                j = reach[np.argmin(D[a, reach])]
                j = pred[a, j]
                while j >= 0 and j != e:
                    drop.append(j)
                    j = pred[a, j]
                drop.append(e)
            K[idx[np.unique(np.asarray(drop, dtype=int))]] = False
        idx = np.flatnonzero(K)
        if len(idx) == 0:
            return {"n_endpoints": 0, "n_junctions": 0, "skeleton_length": 0.0}
        W = self.edge_len[idx][:, idx]
        r = self._runs(K, idx)
        jmask = r >= 3
        n_junc = connected_components(W[jmask][:, jmask], directed=False)[0] if jmask.any() else 0
        return {"n_endpoints": int((r == 1).sum()), "n_junctions": int(n_junc),
                "skeleton_length": float(minimum_spanning_tree(W).sum())}

    @staticmethod
    def _tri_fraction(s):
        """Area fraction of each triangle where the linear interpolant of its vertex values s (m, 3) is > 0."""
        pos = s > 0
        npos = pos.sum(1)
        frac = (npos == 3).astype(float)
        for k in (1, 2):                                 # one vertex on its own side: a corner triangle
            sel = npos == k
            if not sel.any():
                continue
            ss = s[sel]
            lone = np.argmax(pos[sel] if k == 1 else ~pos[sel], axis=1)
            r = np.arange(len(ss))
            a, b, c = ss[r, lone], ss[r, (lone + 1) % 3], ss[r, (lone + 2) % 3]
            corner = (a / (a - b)) * (a / (a - c))
            frac[sel] = corner if k == 1 else 1 - corner
        return frac

    def _superlevel_fraction(self, s):
        """Faces that intersect {s > 0} and the fraction of their area inside it (exact for P1 fields)."""
        sf = s[self.g.F]
        fa = np.flatnonzero(sf.max(1) > 0)
        return fa, self._tri_fraction(sf[fa])

    def spot_defects(self, hi, v, thr) -> dict:
        """Per-component area, perimeter, isoperimetric ratio and Euler characteristic.

        Areas and perimeters are those of the superlevel set {v > thr} of the piecewise-linear field
        (exact per triangle), so a round spot has 4 pi A / P^2 close to 1 at any mesh resolution.
        """
        g = self.g
        idx = np.flatnonzero(hi)
        nc, lab = connected_components(g.adjacency[hi][:, hi], directed=False)
        full = np.full(g.n, -1)
        full[idx] = lab
        size = np.bincount(lab, minlength=nc)
        fa, frac = self._superlevel_fraction(v - thr)
        sf = (v - thr)[g.F[fa]]
        area = np.bincount(full[g.F[fa, np.argmax(sf, axis=1)]], frac * self.face_area[fa], minlength=nc)
        E = g.edges[hi[g.edges[:, 0]] & hi[g.edges[:, 1]]]
        Fin = g.F[hi[g.F].all(1)]
        chi = size - np.bincount(full[E[:, 0]], minlength=nc) + np.bincount(full[Fin[:, 0]], minlength=nc)
        faces, seg = contour_segments(g, v - thr)
        sf = (v - thr)[g.F[faces]]
        owner = full[g.F[faces, np.argmax(sf, axis=1)]]
        perim = np.bincount(owner, seg, minlength=nc)
        keep = size >= self.min_size
        area, perim, chi = area[keep], perim[keep], chi[keep]
        if not len(area):
            return {}
        Q = 4 * np.pi * area / np.maximum(perim, 1e-300) ** 2
        irregular = Q < Q_IRREGULAR
        outlier = np.abs(np.log(area / np.median(area))) > np.log(SIZE_FACTOR)
        nondisk = chi != 1
        return {"spot_irregular_frac": float(irregular.mean()), "spot_size_outlier_frac": float(outlier.mean()),
                "spot_nondisk_frac": float(nondisk.mean()),
                "spot_defect_frac": float((irregular | outlier | nondisk).mean()),
                "spot_Q": Q, "spot_area": area}

    # ---- main entry --------------------------------------------------------------------------------
    def _spectrum(self, f):
        g = self.g
        fc = f - g.mean(f)
        total = g.M @ (fc * fc)
        c = self.PhiTM @ fc
        p = c * c
        share = np.bincount(self.bins, p, minlength=len(KAPPA_EDGES) - 1) / max(total, 1e-300)
        centroid = (self.kappa * p).sum() / max(p.sum(), 1e-300)
        return share, max(1 - p.sum() / max(total, 1e-300), 0.0), centroid

    def describe(self, u, v) -> dict:
        """All features by name, plus the pattern type and (extended) diagnostics that are not features."""
        g = self.g
        u, v = np.asarray(u, float), np.asarray(v, float)
        out = dict.fromkeys(self.names, np.nan)
        q05, q50, q90, q95, q99 = np.quantile(v, [0.05, 0.5, 0.9, 0.95, 0.99])
        out.update(v_q50=q50, v_q90=q90, v_q99=q99, v_max=v.max())
        out.update(zip(("u_q01", "u_q10", "u_q50"), np.quantile(u, [0.01, 0.1, 0.5])))
        if q95 - q05 < 1e-3:
            out["type"] = out["type_detail"] = "uniform"
            return out
        vc = v - g.mean(v)
        out["wavelength"] = 2 * np.pi / np.sqrt(max(-(vc @ (g.L @ vc)) / (g.M @ (vc * vc)), 1e-30))
        thr = 0.5 * (q05 + q95)
        hi = v > thr
        nc, lab = connected_components(g.adjacency[hi][:, hi], directed=False)
        Mh = g.M[hi]
        areas = np.bincount(lab, Mh, minlength=nc)
        sizes = np.bincount(lab, minlength=nc)
        areas = areas[sizes >= self.min_size]
        n = len(areas)
        A = Mh.sum()
        P = _contour_length(g, v - thr)
        E_in = int((hi[g.edges[:, 0]] & hi[g.edges[:, 1]]).sum())
        euler = int(hi.sum() - E_in + int(hi[g.F].all(1).sum()))
        out.update(n_spots=n, area_fraction=A / g.area, compactness=4 * np.pi * A * max(n, 1) / P ** 2 if P else 0,
                   euler_per_spot=euler / max(n, 1), perimeter_density=P / g.area)
        if n:
            rel = areas / g.area
            out.update(size_q10=np.quantile(rel, 0.1), size_q50=np.quantile(rel, 0.5),
                       size_q90=np.quantile(rel, 0.9), size_spread=rel.std() / rel.mean())
        Ml = g.M[~hi]
        out.update(v_in=(Mh @ v[hi]) / A, u_in=(Mh @ u[hi]) / A,
                   v_out=(Ml @ v[~hi]) / Ml.sum(), u_out=(Ml @ u[~hi]) / Ml.sum())
        share, unres, kv = self._spectrum(v)
        out.update(zip(GROUPS["spectrum"][:-3], share))
        out.update(P_unresolved=unres, kappa_v=kv, kappa_u=self._spectrum(u)[2])
        spots = n >= 3 and out["compactness"] > 0.45 and out["area_fraction"] < 0.5
        out["type"] = "spots" if spots else "other"
        out["type_detail"] = out["type"]
        if self.extended:
            out.update(self.orientation(v))
            out.update(self.spot_defects(hi, v, thr))
            sk = self.skeleton_stats(hi, v)
            out.update(sk)
            lr = self.lambda_ref
            out.update(endpoint_density=sk["n_endpoints"] * lr ** 2 / g.area,
                       junction_density=sk["n_junctions"] * lr ** 2 / g.area,
                       skeleton_density=sk["skeleton_length"] * lr / g.area)
            if not spots:     # stripes keep long branches; a maze is dominated by branch points
                out["type_detail"] = "maze" if sk["n_junctions"] > 0.5 * max(sk["n_endpoints"], 1) else "stripes"
        return out

    def __call__(self, u, v) -> tuple[np.ndarray, str]:
        """Feature vector (NaN where undefined) and pattern type for one snapshot."""
        d = self.describe(u, v)
        return np.array([d[n] for n in self.names], dtype=float), d["type"]
