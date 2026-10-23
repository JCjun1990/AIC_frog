# 曲面图灵斑图 · AI-PDE 框架（AI-MECH-2026-002）

> 最新交接（2026-10-23）：见 [HANDOVER.md](HANDOVER.md)——改进前后对比、改动清单、复现命令、注意事项与待办。

本仓库只保留**比赛最终提交所需的内容**：可复现代码、四组正式结果、任务 3 反演、
证据与提交包。研究过程与原始扫描数据已归档在标签
`pre-submission-cleanup-20261022`，不在当前主分支。

## 当前提交内容

```
submission/
  README_提交.md                提交说明（结果、反演、复现、口径、排除项）
  PACKAGE_MANIFEST.json         逐文件来源与 SHA256
  task1_sphere/                 球面斑点 45 斑、七条带条纹（MAT/证书/图/摘要）
  task2_frog/                   青蛙斑点 22 斑、条纹 5 段（MAT/证书/图/摘要）
  task3_inversion/              反演参数、目标—生成对比、误差指标
  evidence/                     评测与校核证据
  code/                         可复现代码 + 两个组委会附件
  task3_inversion/best_match/   反演参数下最接近目标的生成稳态（2026-10-22 晚新增）
  report/初赛技术报告_v3.pdf     两页技术报告（单独提交用；v2 为上一版）
  drafts/待填_初赛_AI-MECH-2026-002_result.zip   当前一致性草稿 ZIP
  待填_初赛_AI-MECH-2026-002_result.zip   历史旧草稿（不要上传）
```

根目录的 `turing_rd/`、`scripts/`、`tests/` 是开发版代码与回归测试；
`frog.mat`、`Solution.mat` 是组委会附件原件。

## 状态与注意

- 提交包与报告已完成工程验收（历史基准；本次验收见 `submission/VALIDATION.md`）。
- **队名仍为“待填”**：ZIP 名、报告队伍行均为占位符；补齐队名并重新生成后方可上传。
- 正式候选（MAT SHA256 前缀）：球面斑点 `44f5931a`、球面条纹 `ef132d82`、
  青蛙斑点 `19a92c65`、青蛙条纹 `aa99f137`。
- 完整残差阈值 1e-11；四组均有三配置有限数值谱证书；青蛙最大已采样实部分别约 −1.0349e-4、−1.5483e-4；
  有限谱检查不是数学全谱证明。
- 研究阶段（adaptive v1/v2）没有超过既有 A2 基线，未合并进本提交版本。

## 2026-10-22 晚：按评分标准的改进

- **任务 3 生成斑图（反演精度 30% 中的「目标与生成斑图相似度」）**：反演参数下模型多稳，原来提交的是一个随机初值
  的稳态（21 斑）。按单快照参数识别「取最接近数据的系综成员」的思路（Kazarnikov 等 2025，arXiv:2504.02530；
  Schnörr & Schnörr 2023），`scripts/inversion_best_match.py` 用 AI 以目标斑点布局为锚生成 66 个初值、Newton 精修，
  在完全稳态且谱稳定的 45 个中取特征距离最小者：16 斑（目标 15），特征距离 J 1.95 → 1.55，交叉点密度与目标一致，
  完整残差 2.0e-14，三组 shift-invert 谱全负。F、k 不变。Windows 与 Mac 选出同一个初值。
  结果在 `submission/task3_inversion/best_match/`（`--install` 装入并登记到清单）。
- **报告 v3**（`scripts/report_v3.py`）：按评分细则要求的条目重写（公式与推导、AI-PDE 架构、数据、量纲、稳定性分析、
  校核、资源、参考文献），去掉内部代号；数字全部从提交记录读取。
- **校验可移植**：`submission_workflow.py verify` 在逐位相同的平台（Windows x86）严格比对；在其他 CPU 上若只有
  离散算子的末位舍入不同，改为核对网格哈希、总质量并重算完整残差（`tests/test_portable_verify.py`）。
- **命名打包**：`i03_package.py --team <队名>` 会去掉 README 中的草稿说明，不再打入内部 VALIDATION.md；
  `i03_verify_package.py` 对命名包检查 named 状态。用测试队名在 Windows 上打包并验收：PASS。
- **泛化误差**（`scripts/generalization_eval.py`，笔记本跑独立测试集 128 个 PDE 稳态）：正式网络 v1 类型准确率
  青蛙 68.8% / 球面 89.1%，加密数据重训的 v2 为 92.2% / 95.3%，κ₀ 误差也下降；物理特征输入（v2p）无额外提升。
  结果在 `submission/evidence/generalization/`；正式结果与证书不变。
- **球面条纹稳健性**（`scripts/sphere_bandcount.py`）：AI 提出 6、7 条带候选，谱检查择稳，10 个随机轴向 10/10 稳定
  （旧的随机相位初值 9/23）。结果在 `submission/evidence/sphere_bandcount/`。
- **端到端效率**（`scripts/active_learning.py e2e --model v2`，笔记本）：青蛙留出点上 v2 初值 + PDE 23/24 完成，
  时间步数中位数 10 875，纯 PDE 25 875（约 2.4 倍；v1 为 1.7 倍）。
- **主动学习**（同一脚本）：在类型边界补 96 个稳态/曲面重训 v3，没有可靠提升（边界不确定性来自多稳态），未采用；
  证据在 `submission/evidence/active_learning/`。最终 AI 网络为 v2（`results/ai_generation_fine/*/model.joblib`）。
- **2026-10-23 复核**：任务 3 再加 180 个候选（共 246 个），提交的 16 斑稳态仍最优；球面斑点改用 Thomson 最小能量布局（`scripts/sphere_spots_layout.py`）为 11/23，不如 CVT 的 14/23，未采用。证据在 `submission/evidence/` 下。
- Windows 完整验收（`DELIVERY_VALIDATION.json`）：95 项单元测试、22 项打包检查全部通过。

## 快速开始

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/verify_solver.py
python scripts/verify_geometry.py
```

当前 ZIP 位于 `submission/drafts/`，已按当前证据重新构建；根下旧 ZIP 原样保留，仅作历史草稿。队名仍未确定。

完整提交包已包含 `submission/code/`，可直接在解压目录中查看/复算；
`submission/README_提交.md` 给出逐任务复现命令与口径说明。

## 任务 4

只在前述总决赛考核，不在本仓库当前范围内。
