# 曲面图灵斑图 · AI-PDE 框架（AI-MECH-2026-002）

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
  report/初赛技术报告.pdf        两页技术报告（单独提交用）
  待填_初赛_AI-MECH-2026-002_result.zip   完整提交包（ZIP）
```

根目录的 `turing_rd/`、`scripts/`、`tests/` 是开发版代码与回归测试；
`frog.mat`、`Solution.mat` 是组委会附件原件。

## 状态与注意

- 提交包与报告已完成工程验收（21/21 独立解包检查）。
- **队名仍为“待填”**：ZIP 名、报告队伍行均为占位符；补齐队名并重新生成后方可上传。
- 正式候选（MAT SHA256 前缀）：球面斑点 `44f5931a`、球面条纹 `ef132d82`、
  青蛙斑点 `19a92c65`、青蛙条纹 `aa99f137`。
- 完整残差阈值 1e-11；球面两组有三配置有限谱检查，青蛙两组为 `not_checked`；
  有限谱检查不是数学全谱证明。
- 研究阶段（adaptive v1/v2）没有超过既有 A2 基线，未合并进本提交版本。

## 快速开始

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
python -m unittest discover -s tests        # 83 项
python scripts/verify_solver.py
python scripts/verify_geometry.py
```

完整提交包已包含 `submission/code/`，可直接在解压目录中查看/复算；
`submission/README_提交.md` 给出逐任务复现命令与口径说明。

## 任务 4

只在前述总决赛考核，不在本仓库当前范围内。
