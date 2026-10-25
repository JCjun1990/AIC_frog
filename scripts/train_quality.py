"""Explicit new quality-filtered training protocol; default is audit only, never retrains v1/v2."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from turing_rd.quality import QUALITY_VERSION, quality_summary, select_training_rows
from turing_rd import experiment_cache as EC


def run(data, out, policy, execute_training=False):
    import generalization_eval as G
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError("quality experiment output exists; original models must not be overwritten")
    if policy not in ("all_endpoints", "complete_steady", "steady_and_spectrum"):
        raise ValueError("choose an explicit new quality policy")
    rows = json.loads(data.read_text(encoding="utf-8"))["rows"]
    selected = select_training_rows(rows, policy)
    if execute_training and not selected:
        raise ValueError("no qualifying evidence; training refused")
    report = {"protocol": QUALITY_VERSION, "policy": policy, "input": EC.file_identity(data),
              "source": EC.file_identity(__file__), "all": quality_summary(rows),
              "selected": quality_summary(selected), "training_executed": False,
              "historical_v1_v2_protocol_changed": False}
    out.mkdir(parents=True, exist_ok=False)
    if execute_training:
        import joblib
        model = G.fit(rows, "plain", quality_policy=policy)
        model["quality_protocol"], model["quality_policy"] = QUALITY_VERSION, policy
        weights = out / f"model_quality_{policy}.joblib"
        joblib.dump(model, weights)
        EC.write_training_provenance(weights, report["input"], report)
        report.update(training_executed=True, model=EC.file_identity(weights))
    with (out / "quality.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quality-policy", choices=("all_endpoints", "complete_steady", "steady_and_spectrum"), required=True)
    ap.add_argument("--execute-training", action="store_true", help="explicitly train NEW weights; default only audits labels")
    args = ap.parse_args()
    print(json.dumps(run(args.data, args.out, args.quality_policy, args.execute_training), indent=2))


if __name__ == "__main__":
    main()
