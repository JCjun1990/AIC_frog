# I02 numerical stability comparison

All values are copied from `stability_certificate.json` files produced by `scripts/i02_stability.py certificate`. Finite Krylov sampling is numerical evidence, not a proof of the full spectrum; no global nonlinear stability is claimed.

| candidate | pattern (n) | I01 full residual | max Re lambda: sigma=1e-3, k=12 | max Re lambda: sigma=0, k=20 | max Re lambda: direct LR, k=8 | non-rotation max Re lambda | reliable positive | growth rate dt=1000 | perturbation | I02 status |
|---|---|---|---|---|---|---|---|---|---|---|
| sphere6 seed2 seven-band candidate | stripes_or_maze (7) | 2.437e-14 | -1.477798e-06 | -1.477798e-06 | -1.477798e-06 | -4.768444e-05 | no | -1.476706e-06 | passed | numerical_check_passed |
| sphere6 six bands (I01 seed 0) | stripes_or_maze (6) | 3.289e-15 | +1.665428e-06 | +1.665428e-06 | +1.665428e-06 | -4.609809e-05 | yes | +1.666814e-06 | passed | unstable_positive_mode |
| sphere6 spots (I01 seed 0) | spots (45) | 8.423e-15 | -6.231004e-06 | -6.231004e-06 | -6.231004e-06 | -5.906735e-05 | no | -6.211672e-06 | passed | numerical_check_passed |

## Certificate details

| candidate | MAT sha256 (first 16) | rotation rank | max eigenpair residual | spectra validated | failure reasons |
|---|---|---|---|---|---|
| sphere6 seed2 seven-band candidate | ef132d82c549a789 | 3 | 1.428e-14 | yes | none |
| sphere6 six bands (I01 seed 0) | 1a903c837104ace4 | 3 | 1.123e-13 | yes | reliable_positive_full_jacobian_eigenvalue |
| sphere6 spots (I01 seed 0) | 44f5931a623d1560 | 3 | 1.729e-13 | yes | none |

## Per-candidate reading

- **sphere6 seed2 seven-band candidate**: every checked eigenvalue is resolved negative; the least damped is -1.477798e-06 (rotation overlap 0.999706), and the non-rotation maximum is -4.768444e-05. The small-amplitude evolution decays at -1.476706e-06 per unit time.
- **sphere6 six bands (I01 seed 0)**: the unprojected operator has a reliable positive eigenvalue +1.665428e-06 (rotation overlap 0.999907 in the shifted solve) while the non-rotation maximum is -4.609809e-05. The small-amplitude evolution of that eigenvector grows at +1.666814e-06 per unit time, so the positive sign is confirmed dynamically and is not solver noise. This is a *discrete* instability of a near-zero rotation-like mode; whether it survives mesh refinement is not established here.
- **sphere6 spots (I01 seed 0)**: every checked eigenvalue is resolved negative; the least damped is -6.231004e-06 (rotation overlap 0.998142), and the non-rotation maximum is -5.906735e-05. The small-amplitude evolution decays at -6.211672e-06 per unit time.

## Reproduction cross-check against the historical record

- **sphere6 seed2 seven-band candidate** (`results\ai_generation\tasks\robustness_jobs\ai_sphere6_0.0540_0.0630_2.pkl`): 7 vs 7 components; wavelength relative difference 5.53e-15; rotation-block eigenvalues max relative difference 1.64e-11; non-rotation maximum -4.768444e-05 vs historical `max_re_lambda` -4.768444e-05 (relative difference 6.18e-13). The regenerated field reproduces the historical record's pattern and its rotation-block eigenvalues. The historical `max_re_lambda` numerically equals the new NON-rotation maximum (not the full maximum), i.e. the legacy stability flag ignored the rotation block entirely; for this candidate that block was negative anyway.

## Scope and limits

- Scope is the unprojected discrete Jacobian `J = Mfull^-1 * weak_J` on the robust sphere6 operator, checked with two shift-invert solves and one direct largest-real-part solve per candidate.
- Rotation overlap is descriptive only. A positive eigenvalue near the rotation subspace is never removed from the decision.
- `mathematical_full_spectrum_proof` is `false` for every candidate: a finite ARPACK sample cannot prove that the whole spectrum lies in the left half plane.
- Mesh-refinement and orientation controls were not run, so the continuum limit of the weak rotation-like modes is not decided here.
