"""I03: select the four formal candidates by hash and bind every piece of evidence.

Selection never trusts a directory name.  Each candidate is located through its
I04 experiment id, then verified against:

* the SHA256 recorded in the I04 evaluation record and in the I01 MAT certificate;
* an independent MAT readback + robust-operator full residual (I01 certificate);
* pattern statistics recomputed from the MAT bytes;
* the I02 stability certificate (sphere) or the I03 frog certificate (frog), whose
  own ``input.mat_sha256`` must equal the MAT hash.

    python scripts/i03_candidates.py --out results/i03_delivery/candidates.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for path in (str(ROOT), str(HERE)):
    if str(path) not in sys.path:
        sys.path.insert(0, path)

from json_safe import write_json_exclusive                           # noqa: E402
from turing_rd import Geometry, GSParams, icosphere, load_frog        # noqa: E402
from turing_rd.diagnostics import pattern_stats                      # noqa: E402
from turing_rd.result_io import (file_sha256, read_mat_fields, validate_certificate)  # noqa: E402
from turing_rd.evaluation import verify_certificate_binding          # noqa: E402

I04_RECORDS = ROOT / "results" / "i04_evaluation" / "evaluation_records.json"
DELIVERY = ROOT / "results" / "i03_delivery"
FROG_CERT = DELIVERY / "frog_stability"

SELECTION = {
    "task1_sphere_spots": dict(
        experiment_id="candidate:sphere6_spots", surface="sphere6", pattern="spots",
        task="task1_sphere", label="单位球面 · 斑点",
        stability="results/i02_stability/certificates/spots/stability_certificate.json"),
    "task1_sphere_stripes": dict(
        experiment_id="candidate:sphere6_stripes_seed2", surface="sphere6", pattern="stripes",
        task="task1_sphere", label="单位球面 · 条纹（七条带 seed2）",
        stability="results/i02_stability/certificates/seed2/stability_certificate.json"),
    "task2_frog_spots": dict(
        experiment_id="candidate:frog_spots", surface="frog", pattern="spots",
        task="task2_frog", label="青蛙曲面 · 斑点",
        stability="results/i03_delivery/frog_stability_v2/frog_spots/frog_stability_certificate.json"),
    "task2_frog_stripes": dict(
        experiment_id="candidate:frog_stripes", surface="frog", pattern="stripes",
        task="task2_frog", label="青蛙曲面 · 条纹（迷宫）",
        stability="results/i03_delivery/frog_stability_v2/frog_stripes/frog_stability_certificate.json"),
}
HISTORICAL = dict(
    experiment_id="candidate:sphere6_stripes_6band", surface="sphere6", pattern="stripes",
    task="task1_sphere", label="历史六条带（不作为正式结果）",
    stability="results/i02_stability/certificates/six_bands/stability_certificate.json")


def load_records():
    document = json.loads(I04_RECORDS.read_text(encoding="utf-8"))
    return {record["experiment_id"]: record for record in document["records"]}


def geometry_for(surface):
    mesh = load_frog(ROOT / "frog.mat") if surface == "frog" else icosphere(6)
    return Geometry(mesh, laplacian="robust", cache_dir=None)


def build(key, spec, records, geometries):
    record = records.get(spec["experiment_id"])
    if record is None:
        raise ValueError(f"{key}: I04 record {spec['experiment_id']} not found")
    mat = ROOT / record["evidence"]["mat"]
    summary_path = ROOT / record["evidence"]["i01_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cert = summary["mat_certificate"]
    params = GSParams(**cert["parameters"])
    geom = geometries[spec["surface"]]
    digest = file_sha256(mat)
    if digest != record["evidence"]["mat_sha256"]:
        raise ValueError(f"{key}: MAT hash differs from the I04 record")
    if digest != cert["mat_sha256"]:
        raise ValueError(f"{key}: MAT hash differs from the I01 certificate")
    fresh = validate_certificate(mat, cert, expected_params=params, expected_seed=summary["seed"],
                                 expected_geometry=geom, tol=1e-11)
    u, v = read_mat_fields(mat, geom.n)
    shape = pattern_stats(geom, v)
    stability_path = ROOT / spec["stability"]
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    if spec["surface"] == "frog":
        bound = stability["input"]["mat_sha256"]
        status = stability["decision"]["status"]
        maximum = stability["decision"]["sampled_maxima"][0]
        evidence_level = stability["input"]["evidence_level"]
        positive = stability["decision"]["reliable_positive_eigenvalues"]
        scope = ("frog finite sample; no rotation symmetry; three shift-invert configurations; "
                 "no perturbation test; unshifted LR not usable")
    else:
        bound = verify_certificate_binding(mat, stability)
        status = stability["decision"]["status"]
        maximum = stability["decision"]["full_max_real_checked"]
        evidence_level = "i02_unprojected_three_configuration_k12_k20_k8_with_eigenpair_residuals"
        positive = stability["decision"]["reliable_positive_eigenvalues"]
        scope = "sphere6 finite sample; rotation-labelled modes included in the decision"
    if bound != digest:
        raise ValueError(f"{key}: stability certificate is bound to a different MAT")
    if not fresh["accepted"]:
        raise ValueError(f"{key}: I01 readback acceptance failed: {fresh['failure_reasons']}")
    return {
        "key": key, "label": spec["label"], "task_dir": spec["task"],
        "experiment_id": spec["experiment_id"],
        "surface": spec["surface"], "pattern": spec["pattern"],
        "mat": str(mat.relative_to(ROOT)),
        "mat_sha256": digest,
        "mat_bytes": mat.stat().st_size,
        "summary": str(summary_path.relative_to(ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "seed": int(summary["seed"]),
        "parameters": params.as_dict(),
        "full_residual": fresh["residual"]["value"],
        "i01_accepted": bool(fresh["accepted"]),
        "structural": {"u_dtype": fresh["structure"]["u_dtype"], "v_dtype": fresh["structure"]["v_dtype"],
                       "u_shape": fresh["structure"]["u_shape"], "v_shape": fresh["structure"]["v_shape"],
                       "physical_range": fresh["physical_range"]["passed"],
                       "float32_roundtrip": [fresh["precision"]["u_float32_roundtrip_exact"],
                                             fresh["precision"]["v_float32_roundtrip_exact"]]},
        "pattern_stats": {"type": shape["type"], "n_components": shape["n_components"],
                          "n_low_components": shape.get("n_low_components"),
                          "wavelength": shape["wavelength"], "area_fraction": shape["area_fraction"],
                          "compactness": shape["compactness"], "v_max": shape["v_max"], "v_min": shape["v_min"]},
        "spectral": {"status": status, "max_real": maximum, "positive_eigenvalues": positive,
                     "evidence_level": evidence_level, "scope": scope,
                     "certificate": str(stability_path.relative_to(ROOT)),
                     "certificate_sha256": file_sha256(stability_path),
                     "mathematical_full_spectrum_proof": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DELIVERY / "candidates.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; nothing is overwritten")
    records = load_records()
    geometries = {"frog": geometry_for("frog"), "sphere6": geometry_for("sphere6")}
    candidates = {key: build(key, spec, records, geometries) for key, spec in SELECTION.items()}
    historical = build("historical_six_band", HISTORICAL, records, geometries)
    document = {
        "schema_version": "aic-frog-i03-candidates-v1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selection_policy": ("candidates are located through I04 experiment ids and accepted only after the "
                             "MAT SHA256 equals both the I04 record and the I01 certificate, the full residual "
                             "is recomputed with the actual robust operator, and the stability certificate binds "
                             "the same MAT hash"),
        "sources": {"i04_records": str(I04_RECORDS.relative_to(ROOT)),
                    "i04_records_sha256": file_sha256(I04_RECORDS)},
        "candidates": candidates,
        "historical_only": historical,
        "checks": {
            "all_hashes_match": True,
            "all_i01_accepted": all(c["i01_accepted"] for c in candidates.values()),
            "all_certificates_bound": True,
            "positive_modes_in_candidates": {k: bool(c["spectral"]["positive_eigenvalues"])
                                             for k, c in candidates.items()},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.out, document)
    for key, candidate in candidates.items():
        print(f"{key}: MAT={candidate['mat_sha256'][:16]} residual={candidate['full_residual']:.6e} "
              f"pattern={candidate['pattern_stats']['type']}/{candidate['pattern_stats']['n_components']} "
              f"spectral={candidate['spectral']['status']} max_real={candidate['spectral']['max_real']:+.6e}")
    print(json.dumps(document["checks"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
