"""Build a fresh portable draft from the self-contained submission, without historical caches."""
from __future__ import annotations
import argparse, datetime, hashlib, json, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEM = "AI-MECH-2026-002"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import submission_workflow as manifest_workflow


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


EVIDENCE_SUFFIXES = {".md", ".csv", ".json", ".png"}
FORMAL_PREFIXES = ("task1_sphere/", "task2_frog/", "task3_inversion/")


def source_payload(source):
    """Keep all registered paths, including code, and discover supported evidence."""
    source = Path(source).resolve()
    previous = load(source / "PACKAGE_MANIFEST.json")
    payload = {entry["path"]: dict(entry) for entry in manifest_workflow.manifest_entries(previous)}
    for relative in manifest_workflow.payload_files(source):
        if relative in payload:
            continue
        path = source / relative
        if relative.startswith("evidence/") and (path.suffix.lower() in EVIDENCE_SUFFIXES or
                (relative.startswith("evidence/methodology/best_match_spectrum/") and path.suffix.lower() == ".npz")):
            payload[relative] = {"path": relative, "source": "submission/" + relative,
                                 "reason": "evaluation and numerical verification evidence",
                                 "sanitized": False, "transformations": []}
        elif relative.startswith("code/"):
            trusted = _runtime_source(source, relative)
            payload[relative] = {"path": relative, "source": trusted.relative_to(ROOT).as_posix(),
                                 "reason": "portable runtime source", "sanitized": False,
                                 "transformations": []}
        else:
            raise ValueError("unregistered payload needs explicit provenance: " + relative)
    if source == (ROOT / "submission").resolve():
        for path in sorted((ROOT / "turing_rd").glob("*.py")):
            relative = "code/turing_rd/" + path.name
            payload.setdefault(relative, {"path": relative, "source": "turing_rd/" + path.name,
                                          "reason": "PDE runtime source", "sanitized": False,
                                          "transformations": []})
    return payload


def _runtime_source(source, relative):
    """Only known packaged runtime files may be refreshed from repository sources."""
    if not relative.startswith("code/"):
        return source / relative
    rel = Path(relative).relative_to("code")
    allowed = (rel.as_posix() in ("requirements.txt", "requirements-delivery.txt", "frog.mat", "Solution.mat")
               or (len(rel.parts) == 2 and rel.parts[0] in ("scripts", "turing_rd") and rel.suffix == ".py"))
    if not allowed:
        raise ValueError("unsupported code source: " + relative)
    path = ROOT / rel
    if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("missing/escaping repository source: " + relative)
    return path


def _write_manifest(path, document):
    """Atomic replacement; a failed serialization never truncates the old manifest."""
    import os
    import tempfile
    text = json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=".manifest-", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sync_source_manifest(source, reference_zip=None, update_reasons=None):
    """Synchronize known sources without blessing unexplained document changes.

    A verified ZIP can establish the provenance of prior wording fixes. Later
    document edits require an explicit per-path reason. Code refreshes come only
    from repository sources, and frozen task files must match their prior hash.
    Historical entry fields survive unchanged or in manifest_updates.
    """
    source = Path(source).resolve()
    path = source / "PACKAGE_MANIFEST.json"
    previous = load(path)
    old = {entry["path"]: entry for entry in manifest_workflow.manifest_entries(previous)}
    missing = sorted(relative for relative in old if not (source / relative).is_file())
    if missing:
        raise ValueError("previously listed file disappeared: " + ", ".join(missing))
    payload = source_payload(source)
    reasons = dict(update_reasons or {})
    unknown = set(reasons) - set(payload)
    if unknown or any(not isinstance(reason, str) or not reason.strip() for reason in reasons.values()):
        raise ValueError("invalid update reasons: " + ", ".join(sorted(unknown)))
    reference = manifest_workflow.zip_manifest_hashes(reference_zip) if reference_zip else {}
    reference_sha = sha(reference_zip) if reference_zip else None
    entries, copies, changed = [], [], []
    for relative, info in sorted(payload.items()):
        target = source / relative
        current = sha(target) if target.is_file() else None
        recorded = old.get(relative)
        frozen = relative.startswith(FORMAL_PREFIXES) or relative in ("code/frog.mat", "code/Solution.mat")
        if frozen and (recorded is None or current != recorded["sha256"]):
            raise ValueError("formal result/model/summary changed: " + relative)
        if recorded and current == recorded["sha256"] and target.stat().st_size != recorded["bytes"]:
            raise ValueError("manifest byte count mismatch: " + relative)
        evidence = None
        if recorded and current != recorded["sha256"]:
            if reference.get(relative) == current:
                evidence = {"reason": "existing workspace bytes verified against the accepted historical ZIP",
                            "reference_zip_sha256": reference_sha, "reference_payload_sha256": current}
            elif relative in reasons:
                evidence = {"reason": reasons[relative]}
            else:
                raise ValueError("unexplained payload change; provide verified reference ZIP or --update-reason: " + relative)
        trusted = _runtime_source(source, relative)
        digest = sha(trusted)
        if trusted != target and digest != current:
            if frozen:
                raise ValueError("formal result/model/summary changed: " + relative)
            evidence = {**(evidence or {}), "reason": "refresh packaged runtime from repository source",
                        "repository_source": trusted.relative_to(ROOT).as_posix(),
                        "repository_source_sha256": digest}
            copies.append((target, trusted.read_bytes(), digest))
        if recorded is not None and digest == recorded["sha256"] and evidence is None:
            entries.append(dict(recorded))
            continue
        entry = dict(info)
        history = list(entry.get("manifest_updates", []))
        if recorded is not None:
            history.append({"previous_entry": {k: v for k, v in recorded.items() if k != "manifest_updates"},
                            **(evidence or {"reason": "refresh derived file size/source metadata"})})
            entry["manifest_updates"] = history
        elif reference.get(relative) == digest:
            entry["manifest_updates"] = [{"reason": "new evidence already present in the accepted historical ZIP",
                                          "reference_zip_sha256": reference_sha, "reference_payload_sha256": digest}]
        entry.update(path=relative, sha256=digest, bytes=trusted.stat().st_size,
                     source=trusted.relative_to(ROOT).as_posix() if trusted.is_relative_to(ROOT) else info["source"],
                     source_sha256=digest)
        entries.append(entry)
        if recorded is not None:
            changed.append(relative)
    document = {**previous, "files": entries, "payload_file_count": len(entries),
                "manifest_scope": "workspace_payload"}
    manifest_workflow.manifest_entries(document)
    # All paths, provenance, reference hashes and frozen assets passed before writes.
    for target, data, digest in copies:
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("source changed during synchronization: " + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if document != previous:
        document["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _write_manifest(path, document)
    checked = manifest_workflow.check_manifest(source)
    return {"manifest": str(path), "files": checked["manifest_files"],
            "added": sorted(set(payload) - set(old)), "updated": changed, "removed": [],
            "refreshed_code": [target.relative_to(source).as_posix() for target, _, _ in copies],
            "manifest_sha256": sha(path)}


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
    manifest_workflow.check_manifest(source)
    previous = load(source / "PACKAGE_MANIFEST.json")
    registered = source_payload(source)
    payload = {relative: (_runtime_source(source, relative), info)
               for relative, info in registered.items() if relative != "README_提交.md"}
    for relative, (trusted, info) in payload.items():
        if sha(trusted) != info["sha256"]:
            raise ValueError("repository source changed; synchronize the source manifest first: " + relative)
    # VALIDATION.md is the team's internal draft log ("not uploadable"); a named build leaves it out.
    if (source / "VALIDATION.md").is_file() and team == "待填":
        payload.setdefault("VALIDATION.md", (source / "VALIDATION.md", registered["VALIDATION.md"]))
    else:
        payload.pop("VALIDATION.md", None)
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
        entry = dict(info)
        history = dict(entry.get("historical_provenance", {}))
        history.setdefault("workspace_manifest_entry", dict(info))
        entry.update(path=relative, source=path.relative_to(ROOT).as_posix(),
                     source_sha256=sha(path), sha256=sha(target), bytes=target.stat().st_size,
                     historical_provenance=history)
        entries.append(entry)
    original_readme = (source / "README_提交.md").read_bytes()
    readme = original_readme.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    transformations = []
    if readme.encode("utf-8") != original_readme:
        transformations.append("normalized README line endings to LF")
    before_heading = readme
    readme = readme.replace("# 待填 · 初赛答案包", f"# {team} · {round_}答案包", 1)
    if readme != before_heading:
        transformations.append("set README team/round heading")
    if team != "待填":
        readme, named_changes = named_readme(readme)
        transformations.extend(named_changes)
    target = package / "README_提交.md"; target.write_text(readme, encoding="utf-8", newline="\n")
    readme_info = dict(registered[target.name])
    history = dict(readme_info.get("historical_provenance", {}))
    history.setdefault("workspace_manifest_entry", dict(readme_info))
    readme_info.update(path=target.name, source="submission/" + target.name,
                       source_sha256=sha(source / target.name), sha256=sha(target), bytes=target.stat().st_size,
                       transformations=list(readme_info.get("transformations", [])) + transformations,
                       historical_provenance=history)
    entries.append(readme_info)
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
                "formal_candidates": formal, "files": entries, "payload_file_count": len(entries),
                "manifest_scope": "package_payload",
                "provenance_note": "certificate input paths/code fingerprints are immutable historical provenance; portable locations are in formal_candidates",
                "historical_zip": {"base_commit": "37a4d25f0aa30cb30be47059715130add0aac5d4",
                                   "sha256": sha(source / "待填_初赛_AI-MECH-2026-002_result.zip") if (source / "待填_初赛_AI-MECH-2026-002_result.zip").is_file() else None,
                                   "retained": "Git history and full local safety backup"}}
    (package / "PACKAGE_MANIFEST.json").write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    manifest_workflow.check_manifest(package)
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None: raise ValueError("ZIP CRC failed")
    manifest_workflow.zip_manifest_hashes(zip_path)
    return package, zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", default="待填")
    parser.add_argument("--round", dest="round_", default="初赛")
    parser.add_argument("--source-submission", type=Path, default=ROOT / "submission")
    parser.add_argument("--out", type=Path, default=ROOT / "results/submission_rebuild")
    parser.add_argument("--sync-source-manifest", action="store_true",
                        help="regenerate submission/PACKAGE_MANIFEST.json from the workspace, then exit")
    parser.add_argument("--reference-zip", type=Path, help="verified historical ZIP tracing existing document changes")
    parser.add_argument("--update-reason", action="append", default=[], metavar="PATH=REASON",
                        help="explicit provenance for a deliberately changed non-formal payload")
    args = parser.parse_args()
    reasons = {}
    for value in args.update_reason:
        if "=" not in value:
            parser.error("--update-reason must be PATH=REASON")
        relative, reason = value.split("=", 1)
        manifest_workflow.manifest_path(relative)
        if relative in reasons:
            parser.error("duplicate --update-reason path")
        reasons[relative] = reason
    if (args.reference_zip or reasons) and not args.sync_source_manifest:
        parser.error("--reference-zip/--update-reason require --sync-source-manifest")
    if args.sync_source_manifest:
        print(json.dumps(sync_source_manifest(args.source_submission, args.reference_zip, reasons), ensure_ascii=False))
        return 0
    package, archive = build(args.source_submission, args.out, args.team, args.round_)
    print(json.dumps({"package": str(package), "zip": str(archive), "zip_sha256": sha(archive)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
