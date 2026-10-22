"""Rebuild descriptive Legacy Evaluation metrics from the bundled 96-row CSV."""
import argparse, csv, hashlib, json, statistics
from pathlib import Path


def rebuild(source, output):
    rows = list(csv.DictReader(Path(source).open(encoding="utf-8")))
    required = {"surface", "pipeline", "converged", "stable", "rel_residual", "solve_s", "point", "seed"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("missing legacy columns")
    identities = [(r["surface"], r["pipeline"], r["point"], r["seed"]) for r in rows]
    if len(set(identities)) != len(rows): raise ValueError("duplicate legacy case")
    records = []
    for surface, pipeline in sorted({(r["surface"],r["pipeline"]) for r in rows}):
        a = [r for r in rows if (r["surface"],r["pipeline"]) == (surface,pipeline)]
        records.append(dict(surface=surface,pipeline=pipeline,denominator=len(a),
            legacy_converged=sum(r["converged"].lower()=="true" for r in a),
            legacy_stable=sum(r["stable"].lower()=="true" for r in a),
            recorded_unprojected_residual_le_1e_11=sum(float(r["rel_residual"])<=1e-11 for r in a),
            shape_match="not_reconstructible",mat_readback="not_available",
            solve_s_median=statistics.median(float(r["solve_s"]) for r in a),
            protocol="historical_csv_not_current_mat_acceptance"))
    with Path(output).open("x", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    return records


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(rebuild(args.csv,args.out),indent=2))
