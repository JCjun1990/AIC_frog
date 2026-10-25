"""Bounded, reproducible I02 stability experiments; never replaces an existing artifact.

Commands: spectrum, lowres, perturb, certificate. Each expensive command runs in
one child process with a hard timeout. All experiment outputs must be below
results/i02_stability. No training, orientation projection or positive-mode removal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:       # importable when loaded by path (tests) too
    sys.path.insert(0, str(HERE))
from json_safe import write_json_exclusive  # noqa: E402

BASE = ROOT / "results" / "i02_stability"
CODE = ("scripts/i02_stability.py", "scripts/json_safe.py", "turing_rd/stability.py",
        "turing_rd/spectral_policy.py",
        "turing_rd/diagnostics.py", "turing_rd/steady.py", "turing_rd/geometry.py",
        "turing_rd/mesh.py", "turing_rd/solver.py", "turing_rd/result_io.py")


def write_json(path, obj):
    """Serialize the whole document before creating the file (never truncate evidence)."""
    return write_json_exclusive(path, obj)


def inputs(args):
    import numpy as np
    from turing_rd import Geometry, GSParams, icosphere
    from turing_rd.result_io import (read_mat_fields, validate_certificate,
                                    file_sha256, geometry_identity)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    c = summary["mat_certificate"]
    surface = summary["surface"]
    if surface != "sphere6":
        raise ValueError("I02 experiments here require the audited sphere6 robust geometry")
    g = Geometry(icosphere(6), laplacian="robust", cache_dir=None)
    p = GSParams(**c["parameters"])
    seed = summary["seed"]
    for field,value in p.as_dict().items():
        if field in summary and summary[field] != value:
            raise ValueError(f"Summary {field} contradicts its MAT certificate")
    fresh = validate_certificate(args.mat, c, expected_params=p, expected_seed=seed,
                                 expected_geometry=g)
    u, v = read_mat_fields(args.mat, g.n)
    meta = {"mat_file": str(args.mat), "mat_sha256": file_sha256(args.mat),
            "summary_sha256": file_sha256(args.summary), "parameters": p.as_dict(),
            "seed": seed, "geometry": geometry_identity(g),
            "full_steady_residual": fresh["residual"], "I01_accepted": fresh["accepted"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": {f: file_sha256(ROOT / f) for f in CODE},
            "runtime": {"python": sys.version, "numpy": np.__version__,
                        **{x: importlib.metadata.version(x) for x in ("scipy", "robust-laplacian")}},
            "mass_convention": "J=Mfull^-1*weak_J; B=sqrt(Mfull)*J/sqrt(Mfull), same eigenvalues",
            "rotation_projection": False}
    return g, p, u, v, meta


def spectrum(args):
    import numpy as np
    from turing_rd.result_io import file_sha256
    from turing_rd.stability import solve_spectrum
    g, p, u, v, meta = inputs(args)
    actual_sigma=None if args.which=="LR" else args.sigma
    print(f"Solving {args.which}, sigma={actual_sigma}, k={args.k}, ncv={args.ncv}", flush=True)
    report, W = solve_spectrum(g, p, u, v, k=args.k,
                               sigma=None if args.which == "LR" else args.sigma,
                               which=args.which, ncv=args.ncv, maxiter=args.maxiter,
                               tol=args.eig_tol, seed=args.rng_seed)
    values = np.array([complex(*pair["eigenvalue"]) for pair in report["pairs"]])
    np.savez_compressed(args.out / "eigenpairs.npz", eigenvalues=values, eigenvectors=W)
    report.update(input=meta, eigenpairs_sha256=file_sha256(args.out / "eigenpairs.npz"))
    write_json(args.out / "spectrum.json", report)
    print(json.dumps({k: report[k] for k in ("solver_converged", "returned_k", "full_max_real",
                                           "all_pairs_validated", "found_reliable_positive")} ), flush=True)


def lowres(args):
    import numpy as np
    import scipy.linalg as la
    import scipy.sparse as sp
    from turing_rd import Geometry, STRIPES, icosphere
    from turing_rd.diagnostics import jacobian
    from turing_rd.steady import weak_jacobian, weak_residual
    from turing_rd.stability import solve_spectrum
    from turing_rd.result_io import file_sha256, geometry_identity
    results = []
    for level in (1, 2):
        g = Geometry(icosphere(level), laplacian="robust", cache_dir=None)
        p = STRIPES
        u = .6 + .12 * np.sin(4 * g.V[:, 2])
        v = .16 + .08 * np.cos(3 * g.V[:, 0])
        M = np.tile(g.M, 2)
        J = jacobian(g, p, u, v)
        Jw = weak_jacobian(g, p, u, v)
        d = sp.diags(M) @ J - Jw
        matrix_error = float(np.max(np.abs(d.data))) if d.nnz else 0.0
        rng = np.random.default_rng(2026)
        z = rng.normal(size=2*g.n)
        h = 1e-6
        fd = (weak_residual(g,p,u+h*z[:g.n],v+h*z[g.n:]) -
              weak_residual(g,p,u-h*z[:g.n],v-h*z[g.n:])) / (2*h*M)
        fd_error = float(np.linalg.norm(fd-J@z)/np.linalg.norm(J@z))
        dense = la.eigvals(J.toarray())
        generalized = la.eigvals(Jw.toarray(), np.diag(M))
        err = float(np.max(np.min(np.abs(dense[:,None]-generalized[None,:]), axis=1)))
        report, W = solve_spectrum(g,p,u,v,k=8,which="LR",sigma=None,ncv=32,tol=1e-11)
        sparse = np.array([complex(*q["eigenvalue"]) for q in report["pairs"]])
        sparse_error = float(np.max(np.min(np.abs(sparse[:,None]-dense[None,:]),axis=1)))
        # Independent 2x2 reaction/diffusion decomposition of a manufactured uniform state.
        uc, vc = .4, .25
        mu = la.eigvalsh(g.L.toarray(),np.diag(g.M))
        block = []
        for m in mu:
            block.extend(la.eigvals([[p.Du*m-vc*vc-p.F,-2*uc*vc],
                                    [vc*vc,p.Dv*m+2*uc*vc-p.F-p.k]]))
        uniform = la.eigvals(jacobian(g,p,np.full(g.n,uc),np.full(g.n,vc)).toarray())
        uniform_error = float(np.max(np.min(np.abs(uniform[:,None]-np.array(block)[None,:]),axis=1)))
        passed = bool(matrix_error<1e-13 and fd_error<1e-7 and err<1e-9 and
                      sparse_error<1e-8 and uniform_error<1e-9 and
                      report["solver_converged"] and report["all_pairs_validated"])
        row = {"level":level,"vertices":g.n,"geometry":geometry_identity(g),
               "mass_identity_absolute_error":matrix_error,"finite_difference_relative_error":fd_error,
               "dense_vs_generalized_max_distance":err,"LR_vs_dense_max_distance":sparse_error,
               "uniform_2x2_vs_dense_max_distance":uniform_error,"passed":passed,
               "dense_max_real":float(dense.real.max()),"LR_report":report,
               "note":"Manufactured states validate the implementation, not physical steady-pattern stability."}
        np.savez_compressed(args.out/f"level{level}_dense.npz",dense=dense,generalized=generalized,
                            uniform=uniform,uniform_2x2=block)
        results.append(row)
        print(f"L{level} n={g.n}: passed={passed}, LR-vs-dense={sparse_error:.3e}",flush=True)
    write_json(args.out/"lowres.json",{"cases":results,"passed":all(r["passed"] for r in results),
                "code_sha256":{f:file_sha256(ROOT/f) for f in CODE}})


def perturbation_report(meta, source_spectrum, pair, amplitudes, time_horizon, resolutions,
                        trajectories, rate_diff, eigenvalue_real, perturbations_sha256):
    """Assemble the perturbation evidence document.

    Every acceptance flag is an explicit Python ``bool``.  The historical failure
    was ``passed = ... and rate_diff < max(1e-9, abs(lam.real)*.03)``: ``lam.real``
    is a NumPy scalar, so ``max`` returned ``numpy.float64`` and the comparison a
    ``numpy.bool_`` that ``json`` cannot encode.  The strict writer in
    ``scripts/json_safe.py`` is the second line of defence; this is the first.
    """
    passed = bool(all(bool(r["supports_eigenvalue_rate"]) and bool(r["physical_ranges_valid"])
                      for r in resolutions) and
                  float(rate_diff) < max(1e-9, abs(float(eigenvalue_real)) * .03))
    return {"input": meta, "source_spectrum": source_spectrum, "eigenpair": pair,
            "amplitudes_joint_RMS": amplitudes, "time": time_horizon,
            "method": ("frozen-J linearly implicit Euler, exact nonlinear robust PDE RHS; "
                       "paired unperturbed control"),
            "linear_method": "backward Euler on the actual fixed unprojected J",
            "resolutions": resolutions, "trajectories": trajectories,
            "half_dt_rate_difference": float(rate_diff), "passed": passed,
            "perturbations_sha256": perturbations_sha256,
            "risks": ["Small-amplitude dynamics only; no final attractor or global nonlinear stability claim.",
                      "Frozen-J diagnostic time stepping is separate from the production IMEX solver."]}


def perturb(args):
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    from turing_rd.diagnostics import jacobian, pattern_stats
    from turing_rd.result_io import file_sha256
    from turing_rd.stability import eigenpair_diagnostics
    g,p,u,v,meta = inputs(args)
    saved = json.loads((args.spectrum/"spectrum.json").read_text(encoding="utf-8"))
    if any(saved["input"][key] != meta[key] for key in ("mat_sha256","geometry","parameters","seed")):
        raise ValueError("Spectrum does not belong to this MAT/operator")
    if saved["eigenpairs_sha256"] != file_sha256(args.spectrum/"eigenpairs.npz"):
        raise ValueError("Eigenvector file hash mismatch")
    with np.load(args.spectrum/"eigenpairs.npz") as a:
        values,W = a["eigenvalues"],a["eigenvectors"]
    index=int(np.argmax(values.real)); lam=values[index]
    if abs(lam.imag)>1e-10:
        raise ValueError("This bounded real-mode perturbation test requires a real eigenpair")
    w=W[:,index]; w=w*np.exp(-1j*np.angle(w[np.argmax(np.abs(w))])); w=w.real
    mass=np.tile(g.M,2); area2=2*g.area
    norm=lambda x: np.sqrt(np.sum(mass[:,None]*x*x,axis=0)/area2) if x.ndim==2 else float(np.sqrt(mass@(x*x)/area2))
    w=w/norm(w); J=jacobian(g,p,u,v)
    pair=eigenpair_diagnostics(J,np.array([lam]),w[:,None],mass)[0]
    if not pair["residual_passed"]:
        raise ValueError("Selected eigenpair failed independent residual verification")
    U=np.concatenate([u,v]); amps=np.array([1e-6,-1e-6,1e-5,-1e-5,0.0])
    final=[]; trajectories=[]; resolutions=[]

    def rhs(X):
        a,b=X[:g.n],X[g.n:]
        r=a*b*b
        return np.vstack([p.Du*(g.L@a)/g.M[:,None]-r+p.F*(1-a),
                          p.Dv*(g.L@b)/g.M[:,None]+r-(p.F+p.k)*b])

    for dt in (args.dt,args.dt/2):
        steps=int(round(args.time/dt))
        if steps<2 or steps>1500 or abs(steps*dt-args.time)>1e-8:
            raise ValueError("Perturbation duration must be a multiple of dt; at most 1500 half-step updates")
        lu=sla.splu((sp.eye(2*g.n,format="csc")-dt*J).tocsc())
        nonlinear=w[:,None]*amps[None,:]
        linear=w.copy(); rows=[]; starts=time.monotonic()
        # Frozen-J linearly implicit Euler: (I-hJ*)delta_next = delta+h(f(U*+delta)-J*delta).
        # For small perturbations the nonlinear remainder is O(delta^2); half-dt is a separate check.
        for it in range(steps+1):
            if it % max(1,steps//30)==0 or it==steps:
                diff=nonlinear[:,:4]-nonlinear[:,4,None]
                proj=(mass*w)@diff/area2
                rows.append({"time":it*dt,"linear_norm":norm(linear),"nonlinear_norm":norm(diff).tolist(),
                             "nonlinear_projection":proj.tolist()})
            if it==steps: break
            forcing=nonlinear+dt*(rhs(U[:,None]+nonlinear)-J@nonlinear)
            combined=lu.solve(np.column_stack([forcing,linear]))
            nonlinear=combined[:,:5]; linear=combined[:,5]
            if not np.all(np.isfinite(nonlinear)) or np.max(np.abs(nonlinear))>.01:
                raise FloatingPointError("Perturbation left the declared small-amplitude budget")
        t=np.array([r["time"] for r in rows]); sel=t>=.25*args.time
        n=np.array([r["nonlinear_norm"] for r in rows]); lin=np.array([r["linear_norm"] for r in rows])
        rates=[float(np.polyfit(t[sel],np.log(n[sel,i]),1)[0]) for i in range(4)]
        linear_rate=float(np.polyfit(t[sel],np.log(lin[sel]),1)[0])
        expected_BE=float(-np.log(1-dt*lam.real)/dt)
        finalstate=U[:,None]+nonlinear
        ranges={"u_min":float(finalstate[:g.n].min()),"u_max":float(finalstate[:g.n].max()),
                "v_min":float(finalstate[g.n:].min()),"v_max":float(finalstate[g.n:].max())}
        row={"dt":dt,"steps":steps,"wall_seconds":time.monotonic()-starts,
             "linear_fitted_rate":linear_rate,"linear_BE_expected_rate":expected_BE,
             "continuous_eigenvalue_real":float(lam.real),"nonlinear_fitted_rates":rates,
             "nonlinear_growth_factors":(n[-1]/n[0]).tolist(),
             "linear_growth_factor":float(lin[-1]/lin[0]),"final_ranges":ranges,
             "final_pattern_statistics":[pattern_stats(g,finalstate[g.n:,i]) for i in range(4)],
             "supports_eigenvalue_rate":bool(abs(linear_rate-expected_BE)<max(1e-11,abs(expected_BE)*.01) and
                         all(abs(r-lam.real)<max(1e-9,abs(lam.real)*.1) for r in rates)),
             "physical_ranges_valid":bool(ranges["u_min"]>=0 and ranges["u_max"]<=1 and ranges["v_min"]>=0)}
        resolutions.append(row); trajectories.append(rows); final.append(nonlinear)
        print(f"dt={dt}, linear rate={linear_rate:.9e}, nonlinear rates={rates}",flush=True)
    np.savez_compressed(args.out/"perturbations.npz",eigenvector=w,amplitudes=amps,
                        final_delta_dt=final[0],final_delta_half_dt=final[1])
    rate_diff=float(np.max(np.abs(np.array(resolutions[0]["nonlinear_fitted_rates"])-
                                          np.array(resolutions[1]["nonlinear_fitted_rates"]))))
    report=perturbation_report(meta,str(args.spectrum),pair,amps.tolist(),args.time,resolutions,
                               trajectories,rate_diff,lam.real,
                               file_sha256(args.out/"perturbations.npz"))
    write_json(args.out/"perturbation.json",report)
    print(json.dumps({"perturbation_passed":report["passed"],
                      "half_dt_rate_difference":report["half_dt_rate_difference"]}),flush=True)


def certificate(args):
    import numpy as np
    from turing_rd.diagnostics import jacobian,pattern_stats
    from turing_rd.result_io import file_sha256
    from turing_rd.stability import aggregate_spectra,eigenpair_diagnostics,rotation_subspace
    g,p,u,v,meta=inputs(args); J=jacobian(g,p,u,v); Q,rot=rotation_subspace(g,u,v)
    reports=[]; artifacts=[]
    for directory in args.runs:
        path=directory/"spectrum.json"
        if not path.exists():
            reports.append({"solver_converged":False,"all_pairs_validated":False,
                            "method":"missing_or_timeout","full_max_real":None,"pairs":[]})
            artifacts.append({"directory":str(directory),"status":"missing_or_timeout"})
            continue
        r=json.loads(path.read_text(encoding="utf-8"))
        if any(r["input"][field]!=meta[field] for field in ("mat_sha256","parameters","seed","geometry")):
            raise ValueError("Spectrum binding mismatch")
        if r["eigenpairs_sha256"]!=file_sha256(directory/"eigenpairs.npz"):
            raise ValueError("Spectrum eigenvector file mismatch")
        with np.load(directory/"eigenpairs.npz") as a:
            fresh=eigenpair_diagnostics(J,a["eigenvalues"],a["eigenvectors"],np.tile(g.M,2),Q)
        if len(fresh)!=len(r["pairs"]): raise ValueError("Eigenpair count mismatch")
        for x,y in zip(fresh,r["pairs"]):
            if x["eigenvalue"]!=y["eigenvalue"] or not np.isclose(
                    x["residual_absolute_M"],y["residual_absolute_M"],rtol=1e-5,atol=1e-13):
                raise ValueError("Independent eigenpair residual mismatch")
        r["pairs"]=fresh
        r["returned_k"]=len(fresh)
        r["solver_converged"]=bool(r["solver_converged"] and len(fresh)==r["requested_k"])
        r["full_max_real"]=max((q["eigenvalue"][0] for q in fresh),default=None)
        r["nonrotation_max_real"]=max((q["eigenvalue"][0] for q in fresh if not q["rotation_like"]),default=None)
        r["all_pairs_validated"]=bool(len(fresh)>0 and all(q["residual_passed"] for q in fresh))
        r["found_reliable_positive"]=any(q["positive_reliable"] for q in fresh)
        reports.append(r); artifacts.append({"directory":str(directory),"spectrum_sha256":file_sha256(path),
                                            "eigenpairs_sha256":r["eigenpairs_sha256"]})
    decision=aggregate_spectra(reports,meta["full_steady_residual"]["value"])
    perturbation=None
    if args.perturbation:
        perturbation=json.loads(args.perturbation.read_text(encoding="utf-8"))
        if any(perturbation["input"][key]!=meta[key] for key in ("mat_sha256","geometry","parameters","seed")):
            raise ValueError("Perturbation MAT/operator/parameters/seed binding mismatch")
        artifact=args.perturbation.parent/"perturbations.npz"
        if perturbation["perturbations_sha256"]!=file_sha256(artifact):
            raise ValueError("Perturbation field hash mismatch")
        perturbation={"file":str(args.perturbation),"sha256":file_sha256(args.perturbation),
                      "passed":perturbation["passed"],"resolutions":perturbation["resolutions"],
                      "half_dt_rate_difference":perturbation["half_dt_rate_difference"]}
    write_json(args.out/"stability_certificate.json",{"schema_version":"aic-frog-stability-v1",
                "scope":"unprojected_discrete_J_finite_numerical_stability_check",
                "input":meta,"rotation":rot,"decision":decision,"spectra":reports,"artifacts":artifacts,
                "perturbation":perturbation,"pattern":pattern_stats(g,v),
                "perturbation_required_for_spectral_check":False,
                "I01_certificate_unchanged":True,
                "mathematical_full_spectrum_proof":False,
                "unverified_risks":["Finite Krylov eigenpair checks do not exhaust the full spectrum.",
                                    "Eigenpair residual is a backward error, not a rigorous forward eigenvalue error bound.",
                                    "No global nonlinear stability or mesh-refinement proof."]})
    print(json.dumps(decision),flush=True)
    return decision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("spectrum","lowres","perturb","certificate"))
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--mat",type=Path)
    parser.add_argument("--summary",type=Path)
    parser.add_argument("--timeout",type=float,default=300)
    parser.add_argument("--k",type=int,default=12)
    parser.add_argument("--sigma",type=float,default=.001)
    parser.add_argument("--which",choices=("LM","LR"),default="LM")
    parser.add_argument("--ncv",type=int,default=48)
    parser.add_argument("--maxiter",type=int,default=3000)
    parser.add_argument("--eig-tol",type=float,default=1e-10)
    parser.add_argument("--rng-seed",type=int,default=2026)
    parser.add_argument("--spectrum",type=Path)
    parser.add_argument("--time",type=float,default=600000)
    parser.add_argument("--dt",type=float,default=2000)
    parser.add_argument("--runs",nargs="+",type=Path,default=[])
    parser.add_argument("--perturbation",type=Path)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.out=args.out.resolve()
    if not args.out.is_relative_to(BASE.resolve()) or args.out==BASE.resolve():
        parser.error("Choose a fresh subdirectory of results/i02_stability")
    if not 0<args.timeout<=600 or not 1<=args.k<=40 or not 4<=args.ncv<=160 or not 1<=args.maxiter<=10000:
        parser.error("Bounded budget required: timeout<=600, k<=40, ncv<=160, maxiter<=10000")
    if args.command!="lowres" and (not args.mat or not args.summary):
        parser.error("--mat and --summary are required")
    for field in ("mat","summary","spectrum","perturbation"):
        if getattr(args,field): setattr(args,field,getattr(args,field).resolve())
    args.runs=[p.resolve() for p in args.runs]
    if args.command=="perturb" and (not args.spectrum or not 0<args.time<=1000000 or not 0<args.dt<=4000):
        parser.error("Perturb requires spectrum and bounded positive time/dt")
    if not args.worker:
        args.out.mkdir(parents=True,exist_ok=False)
        snapshot=args.out/"source_snapshot"
        snapshot.mkdir()
        for name in CODE:
            (snapshot/Path(name).name).write_bytes((ROOT/name).read_bytes())
        command=[sys.executable,"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
        env=os.environ.copy()
        for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            env[key]="1"
        env["PYTHONDONTWRITEBYTECODE"]="1";env["PYTHONIOENCODING"]="utf-8"
        env["MPLCONFIGDIR"]=str(BASE/"runtime_cache"/"matplotlib")
        start=time.monotonic();status="error";code=1
        with (args.out/"run.log").open("x",encoding="utf-8") as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            write_json(args.out/"worker_process.json",{"pid":proc.pid,"command":command,
                         "parent_pid":os.getpid(),"timeout_seconds":args.timeout})
            try:
                code=proc.wait(timeout=args.timeout)
                status="completed" if code==0 else "error"
            except subprocess.TimeoutExpired:
                # Windows venv python.exe is a launcher with a real Python child.
                # Kill only this newly started worker tree, before the launcher exits.
                if os.name=="nt":
                    killed=subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],
                                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
                    log.write("\nTIMEOUT process-tree termination: "+killed.stdout.decode(errors="replace"))
                else:
                    proc.kill()
                proc.wait(timeout=20)
                status="timeout";code=124
        write_json(args.out/"execution.json",{"command":command[:-1],"hard_timeout_seconds":args.timeout,
                       "status":status,"returncode":code,"wall_seconds":time.monotonic()-start,"threads":1})
        print((args.out/"run.log").read_text(encoding="utf-8"),end="")
        print(f"{status}: {args.out}",flush=True)
        raise SystemExit(code)
    sys.path.insert(0,str(ROOT))
    result = globals()[args.command](args)
    if args.command == "certificate":
        raise SystemExit(0 if result["numerical_check_passed"] else 2)


if __name__=="__main__":
    main()
