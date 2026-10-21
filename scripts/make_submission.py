"""Build the preliminary-round answer package from existing results (nothing is recomputed).

    python scripts/make_submission.py              team name placeholder 待填
    python scripts/make_submission.py 队名         once the team name is known

Writes dist/<team>_<round>_AI-MECH-2026-002_result/ and the .zip of the same name. The zip holds the
package contents at its root (rule 六.1: no unrelated nested folders). README_提交.md inside the
package takes its numbers from the result files, so it stays consistent with them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEM_ID = "AI-MECH-2026-002"
AI = ROOT / "results" / "ai_generation"
BASE = ROOT / "results" / "baseline"
INV = ROOT / "results" / "inversion"
VER = ROOT / "results" / "verification"

CODE_SCRIPTS = ["ai_generation.py", "invert_target.py", "run_baseline.py", "report_baselines.py",
                "verify_solver.py", "verify_geometry.py"]
TASK_FILES = ["solution.mat", "ai_initial_v.png", "final_v.png", "summary.json"]
BASELINE_FILES = ["initial_v.png", "field_v.png", "convergence.png", "log.csv", "summary.json"]
INVERSION_FILES = ["summary.md", "summary.json", "solution.mat", "final_vs_target.png", "criteria.png",
                   "landscape.png", "cutoff.png"]
VERIFICATION_FILES = ["geometry.json", "geometry_convergence.png", "solver.json", "time_order.png"]
MODEL_FILES = ["dataset.json", "model.joblib", "cv.json"]


def manifest() -> list[tuple[str, Path]]:
    """(path inside the package, source file)."""
    m = []
    for task, surface, train, base in (("task1_sphere", "sphere6", "sphere5", "sphere"),
                                       ("task2_frog", "frog", "frog", "frog")):
        for pattern in ("spots", "stripes"):
            m += [(f"{task}/ai_pde_{pattern}/{f}", AI / "tasks" / f"{surface}_{pattern}" / f) for f in TASK_FILES]
            m += [(f"{task}/baseline_{pattern}/{f}", BASE / f"{base}_{pattern}" / f) for f in BASELINE_FILES]
        m += [(f"{task}/ai_model/{f}", AI / train / f) for f in MODEL_FILES]
    m += [(f"task3_inversion/{f}", INV / f) for f in INVERSION_FILES]
    m += [("ai_vs_solver/summary.md", AI / "summary.md"), ("ai_vs_solver/evaluation.csv", AI / "evaluation.csv"),
          ("ai_vs_solver/baseline_summary.md", BASE / "summary.md"),
          ("ai_vs_solver/baseline_overview.png", BASE / "overview.png")]
    m += [(f"verification/{f}", VER / f) for f in VERIFICATION_FILES]
    m += [(f"code/{f}", ROOT / f) for f in ("requirements.txt", "frog.mat", "Solution.mat")]
    m += [(f"code/turing_rd/{p.name}", p) for p in sorted((ROOT / "turing_rd").glob("*.py"))]
    m += [(f"code/scripts/{f}", ROOT / "scripts" / f) for f in CODE_SCRIPTS]
    return m


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def e(x) -> str:
    return "—" if x is None else f"{x:.1e}"


def readme(team: str, round_: str) -> str:
    t = {k: load(AI / "tasks" / k / "summary.json")
         for k in ("sphere6_spots", "sphere6_stripes", "frog_spots", "frog_stripes")}
    b = {k: load(BASE / k / "summary.json") for k in ("sphere_spots", "sphere_stripes")}
    cv = {k: load(AI / k / "cv.json")["summary"] for k in ("frog", "sphere5")}
    inv = load(INV / "summary.json")
    F0, k0 = inv["given_spots"]["F"], inv["given_spots"]["k"]
    est, cf, fo = inv["estimate"], inv["closed_form"], inv["features_only"]
    final = [r for r in inv["final"] if r["label"] == "estimate"]
    n_fin = sum(r["rel_residual"] <= 1e-11 and (r["stability"] or {}).get("max_real", 1) < 0 for r in final)

    def row(task, key, label):
        r = t[key]
        stab = f"稳定，max Re λ = {r['max_re_lambda']:.1e}" if r["stable"] else "残差未达标，未判定"
        n = f"{r['n_components']} 个斑点" if r["type"] == "spots" else f"条纹 / 迷宫（{r['n_components']} 段）"
        return (f"| {task} | {label}（F = {r['F']}, k = {r['k']}） | {n} | {e(r['rel_residual'])} | {stab} | "
                f"{r['total_s'] / 60:.0f} 分钟 |")

    def sphere_section():
        sp, st = t["sphere6_spots"], t["sphere6_stripes"]
        if all(r["rel_residual"] <= 1e-11 and r["stable"] for r in (sp, st)):
            return (f"""## 单位球面：转动修正 Newton

- 球面的转动对称只被网格轻微打破，图案会整体极慢地刚体转动，普通 Newton 很难一步转到网格偏好的方向。
- 做法：加入 3 个转动生成元和相位条件，求相对平衡态；再跟随慢转动到网格偏好的朝向，用普通 Newton 精修成离散系统的真稳态。
- 结果：斑点相对残差 {e(sp['rel_residual'])}（扣除转动后 {e(sp.get('rel_residual_rot'))}），条纹 {e(st['rel_residual'])}（扣除转动后 {e(st.get('rel_residual_rot'))}）；线性稳定性已在排除转动模后验证。
- 对照：只用求解器的基线停在 {e(b['sphere_spots']['final_residual']['steady_rel'])}（斑点）和 {e(b['sphere_stripes']['final_residual']['steady_rel'])}（条纹），见 `task1_sphere/baseline_*`。""")
        return (f"""## 单位球面结果尚未完全收敛（如实说明）

- 球面两组的相对残差为 {e(sp['rel_residual'])}（斑点）和 {e(st['rel_residual'])}（条纹），没有达到 1e-11，所以没有给出线性稳定性结论。
- 原因：球面的转动对称只被网格轻微打破，图案会整体极慢地刚体转动；Newton 很难一步转到网格偏好的方向。
- 只用求解器的基线同样停在 {e(b['sphere_spots']['final_residual']['steady_rel'])}（斑点）和 {e(b['sphere_stripes']['final_residual']['steady_rel'])}（条纹），见 `task1_sphere/baseline_*`。""")

    def pct(x, x0):
        return f"{(x - x0) / x0 * 100:+.1f}%"

    return f"""# {team} · {round_}答案包（{PROBLEM_ID}）

赛题：生物曲面扩散动力学及图灵斑图生成、演化与调控。本包含初赛考核的任务 1（单位球面生成）、任务 2（青蛙曲面生成）、任务 3（反演）。技术报告 PDF 另行提交。

每个任务都含 AI 方法，且都由 PDE 求解器校验：
- **任务 1、2（AI 初值 + Newton 精修）**：小型神经网络（scikit-learn MLP 集成）由 (F, k) 预测斑图类型和统计量（斑点密度与剖面、LB 谱环、分位数），据此合成初值；PDE 求解器（曲面 Laplace–Beltrami + SBDF2 积分 + 带线搜索的 Newton）把它精修到稳态，再做线性稳定性检验。训练数据只来自我们在题给网格上自己生成的稳态（每个曲面 45 个参数点），题给参数点不在训练集中。
- **任务 3（特征匹配 + PDE 残差判据，GP 代理）**：求解器在环，按类型、特征波长、方向性、缺陷四类特征和积分恒等式 PDE 残差反演 F、k（Du、Dv 取题给值）。

## 结果一览

相对残差为稳态方程残差的相对值；判定收敛的阈值为 ≤ 1e-11。

| 任务 | 参数 | 结果 | 相对残差 | 线性稳定性 | 用时 |
| --- | --- | --- | --- | --- | --- |
{row("1 单位球面", "sphere6_spots", "斑点")}
{row("1 单位球面", "sphere6_stripes", "条纹")}
{row("2 青蛙", "frog_spots", "斑点")}
{row("2 青蛙", "frog_stripes", "条纹")}

**任务 3 反演**（真值 F = {F0}，k = {k0}）

| 方法 | F | k | F 误差 | k 误差 |
| --- | --- | --- | --- | --- |
| 积分恒等式闭式初值 | {cf['F']:.4f} | {cf['k']:.4f} | {pct(cf['F'], F0)} | {pct(cf['k'], k0)} |
| 仅特征匹配 | {fo['F']:.4f} | {fo['k']:.4f} | {pct(fo['F'], F0)} | {pct(fo['k'], k0)} |
| **特征 + PDE 残差（最终）** | **{est['F']:.4f}** | **{est['k']:.4f}** | {pct(est['F'], F0)} | {pct(est['k'], k0)} |

- 不确定带：F ∈ [{est['band_F'][0]:.4f}, {est['band_F'][1]:.4f}]（含真值），k ∈ [{est['band_k'][0]:.4f}, {est['band_k'][1]:.4f}]（真值在带外 {k0 - est['band_k'][1]:.4f}）。
- 最终参数下 {len(final)} 个随机种子中 {n_fin} 个收敛到相对残差 ≤ 1e-11 且线性稳定，类型均为斑点。
- 目标与生成斑图对比：`task3_inversion/final_vs_target.png`；逐项特征对照、误差指标和残差表：`task3_inversion/summary.md`。

**AI 的效果**（青蛙，8 个留出参数点 × 3 个种子，`ai_vs_solver/summary.md`）
- AI 初值 + Newton：24/24 收敛且稳定；普通初值 + 时间积分：22/24。求解时间中位数 673 s 对 1159 s，约快 1.7 倍。
- 网络的留一法泛化误差（`task*/ai_model/cv.json`）：青蛙类型准确率 {cv['frog']['type_accuracy']:.0%}、波长相对误差中位数 {cv['frog']['kappa0_rel_median']:.1%}；单位球 {cv['sphere5']['type_accuracy']:.0%}、{cv['sphere5']['kappa0_rel_median']:.1%}。

{sphere_section()}

## 目录

```
README_提交.md
task1_sphere/              任务 1：单位球面（icosphere L6，40962 个节点）
  ai_pde_spots/            AI-PDE 结果：solution.mat（Solution.u / Solution.v，与附件格式相同）、
  ai_pde_stripes/            ai_initial_v.png（AI 初值）、final_v.png（终态）、summary.json（残差、Newton 记录、稳定性、用时）
  baseline_spots/          只用求解器的基线：initial_v.png、field_v.png、convergence.png（收敛曲线）、
  baseline_stripes/          log.csv（逐步残差与斑图统计）、summary.json（残差、稳定性特征值、用时）
  ai_model/                网络：dataset.json（自生成训练数据，L5 球面）、model.joblib（训练好的网络）、cv.json（留一法泛化误差）
task2_frog/                任务 2：青蛙曲面（frog.mat，29010 个节点）；子目录同上
task3_inversion/           任务 3：summary.md / summary.json（反演参数、误差、不确定带、残差与稳定性、特征对照）、
                             solution.mat（最终参数下的代表解）、final_vs_target.png（目标与生成对比）、
                             criteria.png（三类判据在 (F, k) 网格上的分布）、landscape.png（目标函数地形）、
                             cutoff.png（预处理低通截断的选择）
ai_vs_solver/              AI-PDE 与普通流程对比：summary.md（成功率、用时、端到端泛化误差）、evaluation.csv（每次运行）、
                             baseline_summary.md、baseline_overview.png（四组基线汇总）
verification/              求解器校核：geometry.json、geometry_convergence.png（LB 算子与曲率的网格收敛）、
                             solver.json、time_order.png（时间收敛阶、梯度检查、后端一致性）
code/                      可复现代码（在此目录下运行）
  turing_rd/               核心包：网格、Laplace–Beltrami、Gray–Scott 求解器、稳态 Newton、诊断、特征、AI 初值、反演组件
  scripts/                 一键脚本（见下）
  frog.mat、Solution.mat   组委会附件，原样
  requirements.txt
```

## 环境

Python 3.11，Mac 或 Linux，CPU 即可（不需要 GPU）：

```bash
cd code
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

测试过的版本见 `requirements.txt` 注释（numpy 2.4、scipy 1.17、torch 2.14、scikit-learn 1.9）。所有脚本固定随机种子。

## 一键复现（在 `code/` 下运行）

结果写到 `code/results/` 下，文件名与本包相同。用时为 Apple M5 实测。

| 内容 | 命令 | 用时 |
| --- | --- | --- |
| 求解器 / 几何校核 | `python scripts/verify_solver.py`，`python scripts/verify_geometry.py` | 几分钟 |
| 四组基线 | `python scripts/run_baseline.py --surface frog --pattern spots`（`--surface sphere`、`--pattern stripes` 同理），汇总 `python scripts/report_baselines.py` | 每组 2–20 分钟 |
| 反演（任务 3） | `python scripts/invert_target.py`（`--quick` 只检查环境，首次运行约 4 分钟） | 约 21 分钟（8 进程） |
| AI-PDE 生成（任务 1、2），从零开始 | `python scripts/ai_generation.py all`：自生成数据 → 训练 → 对比评测 → 任务结果 | 1–2 小时；中断后可续跑 |

只复现任务结果、跳过自生成数据：把附带的网络放回原处，再运行生成阶段（本次实测：青蛙两组共约 10 分钟，球面两组共约 45 分钟）。

```bash
mkdir -p results/ai_generation && cp -R ../task2_frog/ai_model results/ai_generation/frog && cp -R ../task1_sphere/ai_model results/ai_generation/sphere5
python scripts/ai_generation.py generate
```

若 scikit-learn 版本不同导致 `model.joblib` 读不进来，先运行 `python scripts/ai_generation.py train`（每个曲面约 2 分钟，含留一法交叉验证）用附带的 `dataset.json` 重新训练；重训得到的 `cv.json` 与附带的逐位相同。
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("team", nargs="?", default="待填", help="team name (default: placeholder 待填)")
    ap.add_argument("--round", default="初赛", dest="round_")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    args = ap.parse_args()
    team = args.team.strip()
    if not team or any(c in team for c in '/\\:*?"<>|'):
        sys.exit(f"invalid team name: {args.team!r}")
    name = f"{team}_{args.round_}_{PROBLEM_ID}_result"
    out = Path(args.out)
    pkg, zpath = out / name, out / f"{name}.zip"

    m = manifest()
    missing = [str(src.relative_to(ROOT)) for _, src in m if not src.is_file()]
    if missing:
        sys.exit("missing source files:\n  " + "\n  ".join(missing))
    if pkg.exists():
        shutil.rmtree(pkg)
    for rel, src in m:
        dst = pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (pkg / "README_提交.md").write_text(readme(team, args.round_), encoding="utf-8")

    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(pkg.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(pkg).as_posix())

    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
    for i in infos:
        print(f"{i.file_size / 1024:9.1f} KB  {i.filename}")
    raw = sum(i.file_size for i in infos)
    print(f"\n{len(infos)} files, {raw / 2**20:.1f} MB uncompressed, zip {zpath.stat().st_size / 2**20:.1f} MB")
    print(f"package: {pkg}\nzip:     {zpath}")


if __name__ == "__main__":
    main()
