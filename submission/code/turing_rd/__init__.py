"""AI-PDE toolkit for Gray-Scott Turing patterns on curved surfaces.

Layers:
    mesh        -- load frog.mat, build icospheres, triangulate quads, export .ply
    geometry    -- Laplace-Beltrami operator, mass, curvature, LB eigenbasis
    solver      -- differentiable IMEX Gray-Scott solver (SciPy LU or PyTorch CG)
    initial     -- suggested and perturbed initial conditions
    diagnostics -- PDE residual, linear stability, pattern statistics, scales
    viz         -- rendering and convergence plots
"""

from .mesh import Mesh, load_frog, icosphere
from .geometry import Geometry
from .solver import GSParams, GrayScottSolver, SPOTS, STRIPES

__all__ = ["Mesh", "load_frog", "icosphere", "Geometry",
           "GSParams", "GrayScottSolver", "SPOTS", "STRIPES"]
