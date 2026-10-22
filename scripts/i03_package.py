"""I03: build the minimal, complete preliminary-round answer package.

    python scripts/i03_package.py --team 待填 --out results/i03_delivery/package

The package contains only the four formal generation results, the inversion result,
the AI models and training records needed to regenerate the initial guesses, the
stability certificates, the verification records, and the code needed to reproduce
them.  Every included file is listed in ``PACKAGE_MANIFEST.json`` with its source and
SHA256; every excluded category is documented in ``README_提交.md``.

The zip holds the package contents at its root (rule 六.1: no unrelated nested
folders) and is named ``<team>_<round>_AI-MECH-2026-002_result.zip``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "results" / "i03_delivery"
PROBLEM = "AI-MECH-2026-002"
AI = ROOT / "results" / "ai_generation"

CODE_SCRIPTS = [
    ("ai_generation.py", "AI 初值生成、训练与评测一键脚本（任务 1、2）"),
    ("invert_target.py", "任务 3 反演一键脚本（去噪、闭式初值、网格打分、GP 极小）"),
    ("run_baseline.py", "纯 PDE 基线（对照组）"),
    ("report_baselines.py", "基线汇总"),
    ("verify_solver.py", "求解器校核：时间收敛阶、梯度检查、后端一致性"),
    ("verify_geometry.py", "几何校核：LB 算子与曲率的网格收敛"),
    ("recover_float64_results.py", "I01 float64 MAT 恢复与证书生成"),
    ("i02_stability.py", "I02 球面多配置稳定性检查"),
    ("i03_frog_stability.py", "I03 青蛙两组的独立谱检查"),
    ("i04_evaluate.py", "I04 统一评测与统计重算"),
    ("json_safe.py", "严格 JSON 写出（I02/I03/I04 共用）"),
]
EVIDENCE = [
    ("ai_vs_solver_summary.md", "results/ai_generation/summary.md",
     "历史 AI 初值 vs 纯积分评测汇总（8 个留出点 × 3 种子）"),
    ("evaluation.csv", "results/ai_generation/evaluation.csv",
     "上述 96 次评测的逐次记录"),
    ("baseline_summary.md", "results/baseline/summary.md", "四组纯 PDE 基线汇总"),
    ("stability_comparison.md", "results/i02_stability/comparison/comparison.md",
     "三份稳定性证书的横向比较（六条带判不稳定，斑点/七条带通过）"),
    ("verification/geometry.json", "results/verification/geometry.json", "几何/LB 校核记录"),
    ("verification/solver.json", "results/verification/solver.json", "求解器与梯度校核记录"),
    ("verification/time_order.png", "results/verification/time_order.png", "时间收敛阶校核图"),
    ("verification/geometry_convergence.png", "results/verification/geometry_convergence.png",
     "LB 算子网格收敛校核图"),
]
INVERSION = [
    ("summary.md", "反演方法、参数表、不确定带与特征对照"),
    ("summary.json", "反演全过程结构化记录（网格、代理、最终解）"),
    ("solution.mat", "最终估计参数下的代表解（Solution.u / Solution.v）"),
    ("final_vs_target.png", "目标与生成斑图对比（原始、去噪、估计、题给参数）"),
    ("criteria.png", "三类判据在 (F, k) 网格上的分布"),
    ("landscape.png", "GP 代理目标函数地形"),
    ("cutoff.png", "热核低通截断 κ½ 的选择"),
    ("grid_points.csv", "7×7 网格与每点得分记录"),
    ("type_map.csv", "网格上斑图类型分布"),
    ("cutoff_curve.csv", "κ½ 扫描曲线"),
]
MODEL = [("model.joblib", "训练好的小型 MLP（scikit-learn）"),
         ("dataset.json", "自生成训练标签（9×5 网格、每点 1 个初值）"),
         ("cv.json", "留一法泛化记录与逐点误差")]
TASKS = (("task1_sphere", "任务 1 · 单位球面（icosphere L6，40962 顶点）", "sphere5"),
         ("task2_frog", "任务 2 · 青蛙曲面（frog.mat，29010 顶点）", "frog"))


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_tree(value):
    """Replace repository-absolute path strings by relative POSIX paths (JSON payloads only)."""
    if isinstance(value, dict):
        return {key: _portable_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_tree(item) for item in value]
    if isinstance(value, str):
        for prefix in (str(ROOT) + "\\", str(ROOT) + "/"):
            if value.startswith(prefix):
                return value[len(prefix):].replace("\\", "/")
        return value
    return value


def copy_payload(source, target):
    """Copy a file; JSON payloads carrying local absolute paths are rewritten portably."""
    source, target = Path(source), Path(target)
    if target.suffix.lower() == ".json":
        text = source.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            portable = _portable_tree(data)
            if portable != data:
                target.write_text(json.dumps(portable, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8")
                return True, ["repository absolute paths rewritten as relative POSIX paths"]
    shutil.copy2(source, target)
    return False, []


def build_files(candidates):
    files, reasons = [], []
    for key, meta in candidates.items():
        task_dir, sub = meta["task_dir"], meta["pattern"]
        files += [
            (f"{task_dir}/{sub}/solution.mat", ROOT / meta["mat"], "正式生成解（float64，Solution.u/v，与附件同格式）"),
            (f"{task_dir}/{sub}/summary.json", ROOT / meta["summary"],
             "I01 MAT 证书：结构、精度、物理范围、完整残差、SHA256"),
            (f"{task_dir}/{sub}/stability_certificate.json", ROOT / meta["spectral"]["certificate"],
             "数值稳定性证书（有限谱采样，绑定同一 MAT 的 SHA256）"),
            (f"{task_dir}/{sub}/final_v.png", DELIVERY / "figures" / f"{key}.png",
             "终态可视化：由提交 MAT 直接渲染（三视角）"),
        ]
    for task_dir, _, train in TASKS:
        for filename, reason in MODEL:
            files.append((f"{task_dir}/ai_model/{filename}", AI / train / filename, reason))
    for filename, reason in INVERSION:
        files.append((f"task3_inversion/{filename}", ROOT / "results" / "inversion" / filename, reason))
    for dest, source, reason in EVIDENCE:
        files.append((f"evidence/{dest}", ROOT / source, reason))
    files.append(("code/requirements.txt", ROOT / "requirements.txt", "依赖与测试版本"))
    files.append(("code/frog.mat", ROOT / "frog.mat", "组委会附件：青蛙曲面网格（原样）"))
    files.append(("code/Solution.mat", ROOT / "Solution.mat", "组委会附件：目标斑图（原样）"))
    for path in sorted((ROOT / "turing_rd").glob("*.py")):
        files.append((f"code/turing_rd/{path.name}", path, "核心包：网格、LB 算子、求解器、Newton、AI、反演、诊断"))
    for filename, reason in CODE_SCRIPTS:
        files.append((f"code/scripts/{filename}", ROOT / "scripts" / filename, reason))
    return files


def readme(team, round_, candidates, inversion):
    sp, st = candidates["task1_sphere_spots"], candidates["task1_sphere_stripes"]
    fp, ft = candidates["task2_frog_spots"], candidates["task2_frog_stripes"]
    estimate, given = inversion["estimate"], inversion["given_spots"]
    band_F, band_k = estimate["band_F"], estimate["band_k"]
    final = [row for row in inversion["final"] if row["label"] == "estimate"]
    n_ok = sum(1 for row in final if row["rel_residual"] <= 1e-11
               and (row["stability"] or {}).get("max_real", 1) < 0)
    seed2000 = next(row for row in final if row["seed"] == 2000)
    feats = seed2000["features"]
    target = inversion["target_features"]

    def row(label, candidate):
        shape = candidate["pattern_stats"]
        spectral = candidate["spectral"]
        kind = f"{shape['type']} / {shape['n_components']}"
        return (f"| {label} | F={candidate['parameters']['F']:.3f}, k={candidate['parameters']['k']:.3f} | "
                f"{kind} | {shape['wavelength']:.3f} | {candidate['full_residual']:.3e} | "
                f"{spectral['status']} | `{candidate['mat_sha256'][:16]}…` |")

    return f"""# {team} · {round_}答案包（{PROBLEM}）

> **交付状态：草稿。** 队名尚未提供，包名与本文中的 `{team}` 为占位符；补齐队名后需按
> `python scripts/i03_package.py --team <队名>` 重新生成，方可上传。

赛题：生物曲面扩散动力学及图灵斑图生成、演化与调控。本包覆盖初赛考核的
**任务 1（单位球面生成）、任务 2（青蛙曲面生成）、任务 3（参数反演）**；任务 4（力学调控）
只在总决赛考核，不在本包内。技术报告 PDF 按赛题要求在“文档类文件”栏目单独提交，不放入本 ZIP。

## 1. 正式结果

| 任务 | 参数 | 形态 / 数量 | 波长 λ | 完整残差 | 有限谱检查 | MAT SHA256 |
| --- | --- | --- | --- | --- | --- | --- |
{row("任务 1 球面斑点", sp)}
{row("任务 1 球面条纹", st)}
{row("任务 2 青蛙斑点", fp)}
{row("任务 2 青蛙条纹", ft)}

- 完整残差 = √[(RMS_M(R_u)² + RMS_M(R_v)²)/2]/F，使用实际 robust L/M，**不做转动投影**；阈值为 1e-11。
- 谱检查是有限 Krylov 采样（k = 8–20，矩阵维度 5.8–8.2 万），特征对残差已逐对验证；
  **不是整个谱的数学证明**，也不构成全局非线性稳定性结论。
- 球面两解为转到网格偏好朝向后的真离散稳态；青蛙两组用三种 shift-invert 配置
  （σ = 10⁻³、0、10⁻⁵）交叉检查，未发现正实部。无移位最大实部搜索在青蛙刚性算子上不收敛，
  该失败记录保留在 `stability_certificate.json` 的 `unshifted_lr_attempt` 中，且**不用于通过判定**。
- 历史六条带解（seed0）存在 +1.67×10⁻⁶ 的可靠正倾斜模态，已不作为正式结果；条纹正式解为七条带 seed2。

## 2. 任务 3 反演

只反演 F、k（D_u、D_v 取题给值）。结果 **F = {estimate['F']:.5f}，k = {estimate['k']:.5f}**，
相对题给参考值 {given['F']:.3f} / {given['k']:.3f} 的误差为
{(estimate['F'] - given['F']) / given['F'] * 100:+.1f}% / {(estimate['k'] - given['k']) / given['k'] * 100:+.1f}%。

- 不确定带：F ∈ [{band_F[0]:.4f}, {band_F[1]:.4f}]，k ∈ [{band_k[0]:.4f}, {band_k[1]:.4f}]；
  k 的题给值 {given['k']:.3f} 落在带外 {given['k'] - band_k[1]:.4f}，说明估计偏低。
- 最终参数下 {len(final)} 个初值中 {n_ok} 个精修到完整残差 ≤ 1e-11 且谱全负；
  代表解（seed 2000）完整残差 {seed2000['rel_residual']:.1e}、{feats['n_spots']:.0f} 个斑。
- 形态对照：类型一致；相干性、端点/交叉点密度在容差内；但特征波长 {feats['wavelength']:.3f}
  对目标 {target['wavelength']:.3f}（−10%）、κ_v {feats['kappa_v']:.1f} 对 {target['kappa_v']:.1f}、
  缺陷比例 {feats['spot_defect_frac']:.2f} 对 {target['spot_defect_frac']:.2f}，
  **若干尺度与缺陷特征尚未完全匹配**。目标的不规则大斑与偏长波长在题给真值下同样出现。
- 参数定义、参考值与形态对照见 `task3_inversion/summary.md` 与 `summary.json`。

## 3. 目录

```
README_提交.md               本文件
PACKAGE_MANIFEST.json        逐文件来源、SHA256 与保留理由
task1_sphere/               任务 1（40962 顶点）
  spots/ solution.mat, summary.json, stability_certificate.json, final_v.png
  stripes/ 同上
  ai_model/ model.joblib, dataset.json, cv.json
task2_frog/                 任务 2（29010 顶点），结构同任务 1
task3_inversion/            任务 3：反演参数、对比图、判据图、网格记录
evidence/                   评测与校核证据（AI vs 纯积分、稳定性横向比较、求解器/几何校核）
code/                       可复现代码与组委会附件
  turing_rd/                核心包
  scripts/                  一键脚本
  frog.mat, Solution.mat    组委会附件（原样）
  requirements.txt
```

## 4. 环境与复现

Python ≥3.11，CPU 即可（无需 GPU）：

```bash
cd code
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\\Scripts\\pip
```

| 内容 | 命令 | 参考用时 |
| --- | --- | --- |
| 求解器 / 几何校核 | `python scripts/verify_solver.py`、`python scripts/verify_geometry.py` | 几分钟 |
| 四组正式解（已有模型） | 见下：把 `task*/ai_model` 放回 `results/ai_generation/`，再 `python scripts/ai_generation.py generate --generate-surfaces sphere6`（青蛙用 `--generate-surfaces frog`） | 任务 2 约 10 分钟；任务 1 约 45 分钟 |
| 从头训练 + 评测 + 生成 | `python scripts/ai_generation.py all` | 1–2 小时，可续跑 |
| 反演 | `python scripts/invert_target.py` | 约 22 分钟（8 进程） |
| 稳定性复核 | `python scripts/i03_frog_stability.py`；球面见 `scripts/i02_stability.py` | 1–2 分钟/组 |

只复现生成、不重训（把附带的网络放回预期位置）：

```bash
mkdir -p results/ai_generation && cp -R ../task2_frog/ai_model results/ai_generation/frog
cp -R ../task1_sphere/ai_model results/ai_generation/sphere5
python scripts/ai_generation.py generate --generate-surfaces frog sphere6
```

`frog.mat`、`Solution.mat` 已放在 `code/` 下，脚本按仓库根目录读取，不需要改路径。

## 5. 口径与局限（引用数字前请先读）

- **AI 泛化/效率**：青蛙 8 个留出点 × 3 种子，AI 初值 + Newton 收敛 24/24、纯积分 22/24；
  墙钟加速比（solve_s 673 s 对 1159 s）只在旧模型同机同负载的那一组成立，其余场景只比较与机器无关的
  积分步数。L5 评测未做 orient 时只有相对平衡态，不能当作完整离散稳态。
- **网络留一法类型准确率**（青蛙 64%、球面 89%）是分类指标，不等于最终斑图成功率；
  新旧模型的两组留一法网格密度不同，不能直接比。
- **多种子稳健性**（历史协议，见 `evidence/i04_stability_comparison.md` 与报告）：
  青蛙斑点 31/33；青蛙条纹收敛 33/33 但目标条纹形态 31/33；球面斑点 17/23 完整稳态、14/23 无正模态；
  球面条纹 22/23 完整稳态、9/23 无正模态。这些是历史多种子实验，不是对单个提交文件的验收。
- 所有 AI 输出都经过 PDE 复算；没有使用外部数据或预训练生成模型直接出结果。

## 6. 未包含内容及依据

| 排除项 | 依据 |
| --- | --- |
| `results/ai_generation_fine/`（加密数据新模型） | 未用于正式结果；体积大（模型 3.4 MB×2、数据 1.4 MB） |
| 训练/评测日志（`run.log`、`stdout_*.log`、`robust_*.log`） | 过程日志冗余，且含机器名与运行环境细节 |
| 旧实验、失效候选（六条带 seed0、L5 早期结果、`baseline/*.ply` 等） | 已被正式候选取代或与结论无关 |
| `results/i04_evaluation/` 全量记录（2.6 MB） | 本包只放横向比较摘要；逐条记录留在仓库 |
| `docs/` 历史版本报告与大纲 | 非本轮提交物；报告按赛题单独上传 |
| `.venv/`、`__pycache__`、`dist/`、`.git/` | 环境/缓存/版本库，不属于答案文件 |
| 本机绝对路径、密钥、个人配置 | 包内 JSON 中仅作说明的仓库路径已改写为相对路径；所有哈希绑定字段（MAT/证书/清单）保持原值，不含任何本机路径、令牌或个人配置 |
| 任务 4 力学调控 | 赛题规定初赛不考核（总决赛内容） |

## 7. 完整性

`PACKAGE_MANIFEST.json` 列出每个文件的包内路径、来源、源 SHA256、包内 SHA256、字节数、保留理由与是否做过路径改写；
`python scripts/i03_verify_package.py <zip>` 会在仓库外独立解包并复核 CRC、清单、MAT 结构/精度/
完整残差、证书哈希绑定、反演一致性、代码与路径卫生。逐文件校验值以该清单为准。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", default="待填")
    parser.add_argument("--round", default="初赛", dest="round_")
    parser.add_argument("--out", type=Path, default=DELIVERY / "package")
    args = parser.parse_args()
    team = args.team.strip()
    if not team or any(character in team for character in '/\\:*?"<>|'):
        sys.exit(f"invalid team name: {args.team!r}")
    name = f"{team}_{args.round_}_{PROBLEM}_result"
    args.out = args.out.resolve()
    package, zip_path = args.out / name, args.out / f"{name}.zip"
    if package.exists() or zip_path.exists():
        sys.exit("package already exists; nothing is overwritten")
    candidates = load("results/i03_delivery/candidates.json")["candidates"]
    inversion = load("results/inversion/summary.json")
    files = build_files(candidates)
    missing = [(dest, source) for dest, source, _ in files if not Path(source).is_file()]
    if missing:
        sys.exit("missing sources:\n  " + "\n  ".join(f"{d} <- {s}" for d, s in missing))

    package.mkdir(parents=True, exist_ok=False)
    manifest = []
    for dest, source, reason in files:
        target = package / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        sanitized, transformations = copy_payload(source, target)
        manifest.append({"path": dest, "source": str(Path(source).resolve().relative_to(ROOT)),
                         "source_sha256": sha(source), "sha256": sha(target),
                         "bytes": target.stat().st_size, "reason": reason,
                         "sanitized": sanitized, "transformations": transformations})
    checks = {"all_hashes_match": True}
    readme_path = package / "README_提交.md"
    readme_path.write_text(readme(team, args.round_, candidates, inversion), encoding="utf-8")
    manifest.append({"path": "README_提交.md", "source": "generated by scripts/i03_package.py",
                     "source_sha256": None, "sha256": sha(readme_path),
                     "bytes": readme_path.stat().st_size,
                     "reason": "提交说明：结果、反演、复现命令、口径与未包含内容",
                     "sanitized": False, "transformations": []})
    document = {
        "schema_version": "aic-frog-i03-package-v1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "problem": PROBLEM, "round": args.round_, "team": team,
        "delivery_status": "draft: team name pending; regenerate before upload",
        "formal_candidates": {key: {"mat": value["mat"], "mat_sha256": value["mat_sha256"],
                                    "summary": value["summary"], "summary_sha256": value["summary_sha256"],
                                    "stability_certificate": value["spectral"]["certificate"],
                                    "stability_certificate_sha256": value["spectral"]["certificate_sha256"],
                                    "full_residual": value["full_residual"],
                                    "spectral_status": value["spectral"]["status"],
                                    "spectral_max_real": value["spectral"]["max_real"]}
                              for key, value in candidates.items()},
        "inversion": {"estimate": inversion["estimate"], "given": inversion["given_spots"]},
        "files": manifest,
        "exclusions": [
            "results/ai_generation_fine (unused new model)", "process logs", "superseded candidates",
            "full I04 record set", "history reports and outline", "environment/caches/git",
            "absolute local paths and personal configuration", "task 4 (not in preliminary round)",
        ],
    }
    (package / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        infos = archive.infolist()
    if bad is not None:
        sys.exit(f"zip CRC failure at {bad}")
    raw = sum(info.file_size for info in infos)
    print(f"{len(infos)} files, {raw / 2**20:.1f} MB uncompressed, "
          f"zip {zip_path.stat().st_size / 2**20:.1f} MB")
    print(f"package: {package}\nzip:     {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
