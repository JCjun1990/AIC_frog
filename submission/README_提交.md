# 待填 · 初赛答案包（AI-MECH-2026-002）

> **交付状态：草稿。** 队名尚未提供，包名与本文中的 `待填` 为占位符；补齐队名后需按
> `python scripts/i03_package.py --team <队名>` 重新生成，方可上传。

赛题：生物曲面扩散动力学及图灵斑图生成、演化与调控。本包覆盖初赛考核的
**任务 1（单位球面生成）、任务 2（青蛙曲面生成）、任务 3（参数反演）**；任务 4（力学调控）
只在总决赛考核，不在本包内。技术报告 PDF 按赛题要求在“文档类文件”栏目单独提交，不放入本 ZIP。

## 1. 正式结果

| 任务 | 参数 | 形态 / 数量 | 波长 λ | 完整残差 | 有限谱检查 | MAT SHA256 |
| --- | --- | --- | --- | --- | --- | --- |
| 任务 1 球面斑点 | F=0.028, k=0.062 | spots / 45 | 0.475 | 8.423e-15 | numerical_check_passed | `44f5931a623d1560…` |
| 任务 1 球面条纹 | F=0.054, k=0.063 | stripes_or_maze / 7 | 0.471 | 2.437e-14 | numerical_check_passed | `ef132d82c549a789…` |
| 任务 2 青蛙斑点 | F=0.028, k=0.062 | spots / 22 | 0.464 | 2.171e-14 | numerical_check_passed | `19a92c65ec759c9a…` |
| 任务 2 青蛙条纹 | F=0.054, k=0.063 | stripes_or_maze / 5 | 0.483 | 1.006e-14 | numerical_check_passed | `aa99f1373d76cc70…` |

- 完整残差 = √[(RMS_M(R_u)² + RMS_M(R_v)²)/2]/F，使用实际 robust L/M，**不做转动投影**；阈值为 1e-11。
- 谱检查是有限 Krylov 采样（k = 8–20，矩阵维度 5.8–8.2 万），特征对残差已逐对验证；
  **不是整个谱的数学证明**，也不构成全局非线性稳定性结论。
- 球面两解为转到网格偏好朝向后的真离散稳态；青蛙两组用三种 shift-invert 配置
  （σ = 10⁻³、0、10⁻⁵）交叉检查，未发现正实部。无移位最大实部搜索在青蛙刚性算子上不收敛，
  该失败记录保留在 `stability_certificate.json` 的 `unshifted_lr_attempt` 中，且**不用于通过判定**。
- 历史六条带解（seed0）存在 +1.67×10⁻⁶ 的可靠正倾斜模态，已不作为正式结果；条纹正式解为七条带 seed2。

## 2. 任务 3 反演

只反演 F、k（D_u、D_v 取题给值）。结果 **F = 0.02771，k = 0.06004**，
相对题给参考值 0.028 / 0.062 的误差为
-1.0% / -3.2%。

- 不确定带：F ∈ [0.0267, 0.0287]，k ∈ [0.0598, 0.0613]；
  k 的题给值 0.062 落在带外 0.0007，说明估计偏低。
- 最终参数下 6 个初值中 5 个精修到完整残差 ≤ 1e-11 且谱全负；
  代表解（seed 2000）完整残差 8.7e-14、21 个斑。
- 形态对照：类型一致；相干性、端点/交叉点密度在容差内；但特征波长 0.491
  对目标 0.547（−10%）、κ_v 12.6 对 10.6、
  缺陷比例 0.29 对 0.60，
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
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
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
