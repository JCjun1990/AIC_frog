"""Build a fresh portable draft from the self-contained submission, without historical caches."""
from __future__ import annotations
import argparse, datetime, hashlib, json, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = "AI-MECH-2026-002"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def named_readme(readme):
    """For a named build: drop the draft notice (leading '>' block) and the repository-only
    '当前报告' section, which describe the unnamed draft and paths outside the ZIP."""
    lines, out, skip, notice = readme.split("\n"), [], False, False
    for line in lines:
        if line.startswith("> **交付状态：草稿。**"):
            notice = True
        if notice:
            if line.startswith(">"):
                continue
            notice = False
        if line.startswith("## "):
            skip = line.strip() == "## 当前报告"
        if not skip:
            out.append(line)
    text = "\n".join(out).rstrip("\n") + "\n"
    text = text.replace("本次实测环境和模型加载检查见 `VALIDATION.md`。", "")      # internal log is not packaged
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    if "待填" in text or "草稿" in text.split("## 1.")[0]:
        raise ValueError("named README still contains draft placeholders")
    return text, ["removed draft notice", "removed repository-only section '当前报告'"]


def build(source, output, team="待填", round_="初赛"):
    """Never overwrite an existing build; formal MAT/certificates are copied byte-for-byte."""
    source, output = Path(source).resolve(), Path(output).resolve()
    for value in (team, round_):
        if not value.strip() or any(c in value for c in '/\\:*?"<>|'):
            raise ValueError("invalid team/round name")
    name = f"{team}_{round_}_{PROBLEM}_result"
    package, zip_path = output / name, output / f"{name}.zip"
    if package.exists() or zip_path.exists():
        raise FileExistsError("package already exists; nothing is overwritten")
    previous = load(source / "PACKAGE_MANIFEST.json")
    # Only the enumerated payload is inherited. ZIPs, build folders and the separate PDF never recurse.
    payload = {r["path"]: (source / r["path"], r) for r in previous["files"]
               if not r["path"].startswith("code/") and r["path"] != "README_提交.md"}
    for path in sorted((source / "evidence").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".csv", ".json", ".png"}:
            relative = path.relative_to(source).as_posix()
            payload[relative] = (path, {"reason": "evaluation and numerical verification evidence"})
    # VALIDATION.md is the team's internal draft log ("not uploadable"); a named build leaves it out.
    if (source / "VALIDATION.md").is_file() and team == "待填":
        payload["VALIDATION.md"] = (source / "VALIDATION.md", {"reason": "current engineering validation and explicit expensive-step limits"})
    else:
        payload.pop("VALIDATION.md", None)
    for path in sorted((ROOT / "turing_rd").glob("*.py")):
        payload["code/turing_rd/" + path.name] = (path, {"reason": "unchanged PDE core source"})
    old_scripts = [r["path"].split("/")[-1] for r in previous["files"] if r["path"].startswith("code/scripts/")]
    for filename in sorted(set(old_scripts + ["submission_workflow.py", "i03_verify_package.py", "i02_generate_candidate.py", "rebuild_legacy_metrics.py",
                                         "inversion_best_match.py", "i03_frog_stability.py", "generalization_eval.py",
                                         "sphere_bandcount.py", "active_learning.py",
                                         "sphere_spots_layout.py"])):
        payload["code/scripts/" + filename] = (ROOT / "scripts" / filename, {"reason": "portable reproduction/verification entry point"})
    for filename in ("requirements.txt", "requirements-delivery.txt", "frog.mat", "Solution.mat"):
        payload["code/" + filename] = (ROOT / filename, {"reason": "runtime requirements or original competition input"})
    for relative, (path, _) in payload.items():
        if not (source / relative).resolve().is_relative_to(source):
            raise ValueError("unsafe source manifest path")
        if not path.is_file():
            raise FileNotFoundError(path)
    # Frozen formal artifacts must still equal the prior manifest before copying.
    old = {r["path"]: r for r in previous["files"]}
    for relative, (path, _) in payload.items():
        if relative.startswith(("task1_sphere/", "task2_frog/", "task3_inversion/")) and relative in old:
            if sha(path) != old[relative]["sha256"]:
                raise ValueError("formal result/model/summary changed: " + relative)
    package.mkdir(parents=True, exist_ok=False)
    entries = []
    for relative, (path, info) in sorted(payload.items()):
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        entries.append({"path": relative, "source": path.relative_to(ROOT).as_posix(),
                        "source_sha256": sha(path), "sha256": sha(target), "bytes": target.stat().st_size,
                        "reason": info.get("reason", "retained formal payload"), "sanitized": False,
                        "transformations": [], "historical_provenance": {k: info[k] for k in ("source", "source_sha256", "sanitized", "transformations") if k in info}})
    readme = (source / "README_提交.md").read_text(encoding="utf-8")
    readme = readme.replace("# 待填 · 初赛答案包", f"# {team} · {round_}答案包", 1)
    transformations = []
    if team != "待填":
        readme, transformations = named_readme(readme)
    target = package / "README_提交.md"; target.write_text(readme, encoding="utf-8", newline="\n")
    entries.append({"path": target.name, "source": "submission/README_提交.md", "source_sha256": sha(source / target.name),
                    "sha256": sha(target), "bytes": target.stat().st_size, "reason": "current portable instructions and evidence limits",
                    "sanitized": False, "transformations": transformations})
    formal = {}
    for task in ("task1_sphere/spots", "task1_sphere/stripes", "task2_frog/spots", "task2_frog/stripes"):
        directory = package / task; summary = load(directory / "summary.json"); spectral = load(directory / "stability_certificate.json")
        cert = summary["mat_certificate"]; decision = spectral["decision"]
        maximum = decision.get("full_max_real_checked")
        if maximum is None: maximum = max(decision["sampled_maxima"])
        formal[task.replace("/", "_")] = {
            "mat": task + "/solution.mat", "mat_sha256": sha(directory / "solution.mat"),
            "summary": task + "/summary.json", "summary_sha256": sha(directory / "summary.json"),
            "stability_certificate": task + "/stability_certificate.json",
            "stability_certificate_sha256": sha(directory / "stability_certificate.json"),
            "full_residual": cert["residual"]["value"], "spectral_status": decision["status"], "spectral_max_real": maximum}
    document = {**{k: previous[k] for k in ("problem", "inversion", "exclusions")}, "schema_version": "aic-frog-i03-package-v2",
                "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "round": round_, "team": team,
                "delivery_status": "draft: team name pending; do not upload" if team == "待填" else "named candidate: separate final report/team check required",
                "formal_candidates": formal, "files": entries,
                "provenance_note": "certificate input paths/code fingerprints are immutable historical provenance; portable locations are in formal_candidates",
                "historical_zip": {"base_commit": "37a4d25f0aa30cb30be47059715130add0aac5d4",
                                   "sha256": sha(source / "待填_初赛_AI-MECH-2026-002_result.zip") if (source / "待填_初赛_AI-MECH-2026-002_result.zip").is_file() else None,
                                   "retained": "Git history and full local safety backup"}}
    (package / "PACKAGE_MANIFEST.json").write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None: raise ValueError("ZIP CRC failed")
    return package, zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", default="待填")
    parser.add_argument("--round", dest="round_", default="初赛")
    parser.add_argument("--source-submission", type=Path, default=ROOT / "submission")
    parser.add_argument("--out", type=Path, default=ROOT / "results/submission_rebuild")
    args = parser.parse_args()
    package, archive = build(args.source_submission, args.out, args.team, args.round_)
    print(json.dumps({"package": str(package), "zip": str(archive), "zip_sha256": sha(archive)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
