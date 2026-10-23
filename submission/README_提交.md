# 待填 · 初赛答案包（AI-MECH-2026-002）

> **交付状态：草稿。** 队名尚未提供，包名与本文中的 `待填` 为占位符；在仓库根目录补齐队名、更新报告队伍行并重新验收后，按
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

- 目标函数容差带（启发式，不是经覆盖率校准的置信区间）：F ∈ [0.0267, 0.0287]，k ∈ [0.0598, 0.0613]；
  k 的题给值 0.062 落在带外 0.0007，说明估计偏低。
- 最终参数下 6 个随机初值中 5 个精修到完整残差 ≤ 1e-11 且历史单配置已采样谱为负；
  随机初值代表解（seed 2000，`task3_inversion/solution.mat`）完整残差 8.7e-14、21 个斑，保留作对照。
- **生成斑图（`task3_inversion/best_match/`）**：反演参数下模型多稳，随机初值会得到 18–26 个斑。
  参照单快照参数识别中「取最接近数据的系综成员」的做法（Kazarnikov 等 2025，arXiv:2504.02530；
  Schnörr & Schnörr 2023 的初值无关特征），AI 以目标斑点布局为锚、用网络预测的稳态斑点剖面生成
  66 个初值（斑点数 15–20），全部由 Newton 精修；45 个达到完整稳态且单移位谱为负，
  取特征距离 J 最小者：**16 斑**（目标 15），完整残差 2.0e-14，三组 shift-invert 谱全部为负
  （最大 −1.86e-4）。F、k 不因此改变。Windows 与 Mac 上同一命令选出同一个初值（J 一致到 6 位）。
- 形态对照（同一测量算子，目标 / 新生成 / 旧代表解）：斑点数 15 / 16 / 21；谱峰波长
  0.542 / 0.498（−8%）/ 0.487（−10%）；方向相干度 0.31 / 0.35 / 0.30；交叉点密度 0.22 / 0.22 / 0.14；
  骨架密度 0.61 / 0.53 / 0.39；特征距离 J 0 / **1.55** / 1.95；与目标逐点相关 1 / 0.25 / 0.15。
- 剩余偏差：目标严格说不是该模型的稳态（v < 0 区域违反最大值原理，以目标为初值在任何参数下都会漂移），
  生成稳态的斑点更规整，波长仍偏短约 8%，缺陷斑比例 0.31 对 0.60。
- 复现：`python code/scripts/inversion_best_match.py`（66 个初值，Windows 20 进程约 38 分钟）。
- 扩大复核（`evidence/best_match_search/`）：另加 180 个不同随机种子的均匀布局候选，最小 J 为 1.673 / 1.728，共 246 个候选中提交的 16 斑稳态（J = 1.554）仍最接近目标。
- 参数定义、参考值与形态对照见 `task3_inversion/summary.md`、`summary.json` 与 `best_match/summary.md`。

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
  best_match/               反演参数下最接近目标的生成稳态（MAT、证书、谱检查、对比图、全部初值记录）
evidence/                   评测与校核证据（AI vs 纯积分、稳定性横向比较、求解器/几何校核）
  generalization/           独立测试集上的泛化误差（v1 / v2 / v2p）、测试集与加密训练数据
  sphere_bandcount/         球面条纹：AI 条带数候选 + 谱检查择稳
  active_learning/          主动学习的负结果与 v2 的端到端效率
  best_match_search/        任务 3 扩大到 246 个候选的复核
  sphere_spots_layout/      球面斑点 Thomson 布局对照（负结果）
code/                       可复现代码与组委会附件
  turing_rd/                核心包
  scripts/                  一键脚本
  frog.mat, Solution.mat    组委会附件（原样）
  requirements.txt
```

## 4. 环境与复现

在 ZIP 解压根目录执行以下命令。Python 3.11/3.12，CPU 即可。
下面的 `python` 必须指向已安装依赖的解释器；安装命令仅供复现者自行选择，本次修复未安装依赖。

```bash
python -m venv .venv
# Linux/macOS:
.venv/bin/python -m pip install -r code/requirements.txt
# Windows PowerShell:
.venv\Scripts\python.exe -m pip install -r code/requirements.txt
```

使用环境内解释器后，以下命令在 Windows/Linux 共用：

```bash
python code/scripts/submission_workflow.py verify --submission .
python code/scripts/submission_workflow.py prepare-models --submission . --out reproduction_inputs
python code/scripts/ai_generation.py generate --out reproduction_inputs --generate-out reproduction_runs/tasks --generate-surfaces frog sphere6
```

`prepare-models` 将网络复制为 `reproduction_inputs/frog/model.joblib` 与
`reproduction_inputs/sphere5/model.joblib`，不依赖旧 `results/` 或 shell 复制命令。
输出目录存在时拒绝覆盖。生成步骤较贵，本次未重新训练或重跑完整生成。
**此命令重跑 seed0 通用生成流程，不保证找到正式七条带 seed2 或逐位复现既有 MAT。**
正式提交的是包内经证书绑定的四组 MAT；精确回读验收用第一条命令。

独立重算一个有限谱样本（可选，耗时，不能取代已保存的三配置证据）：

```bash
python code/scripts/submission_workflow.py spectrum --submission . --task task2_frog/spots --out reproduction_runs/frog_spots_spectrum
```

谱命令先验收同一 MAT 的完整 robust PDE 残差，再使用未投影 Jacobian 求谱。
`i03_frog_stability.py` 旧默认入口、I04 全量统计及历史 seed2 再生成依赖归档输入，
不属于精简包的即用命令。完整历史流程见 Git 标签 `pre-submission-cleanup-20261022`。
包内 `frog.mat`、`Solution.mat` 原件位于 `code/`；无需本地历史结果目录。
依赖清单中的历史版本注释是原实验环境，不保证任意新 scikit-learn 的 pickle 兼容；
本次实测环境和模型加载检查见 `VALIDATION.md`。跨版本警告应保留。

## 5. 口径与局限（引用数字前请先读）

- **Legacy Evaluation（历史口径，非当前联合成功率）**：青蛙 8 个留出点 × 3 种子，历史旧判据 AI 24/24、纯积分 22/24；不同时代表目标形态匹配、当前完整稳态与三配置谱验收。
  墙钟加速比（solve_s 673 s 对 1159 s）只在旧模型同机同负载的那一组成立，其余场景只比较与机器无关的
  积分步数。L5 评测未做 orient 时只有相对平衡态，不能当作完整离散稳态。
- **泛化误差（独立测试集，`evidence/generalization/`）**：训练区域内 32 个新参数点 × 2 个随机初值（每曲面 64 个 PDE 稳态，
  种子与训练集不重合）。正式结果所用网络 v1 类型准确率青蛙 68.8%、球面 89.1%；同一方法用加密自生成数据（17×9×2，306 态）
  重训的 v2 为 92.2%、95.3%（按参数点含多稳：93.8%、100%），κ₀ 误差 3.4%→2.7%、2.2%→1.7%。加物理特征的 v2p 无额外提升。
  正式结果仍由 v1 生成，证书不变。复算：`python code/scripts/generalization_eval.py evaluate`。
- **端到端效率与主动学习（`evidence/active_learning/`）**：青蛙 8 个留出参数点 × 3 种子，时间步数中位数
  纯 PDE 25 875（22/24 完成）、AI v1 15 062（24/24）、AI v2 10 875（23/24，约 2.4 倍）。在类型边界做不确定性采样
  （每曲面 96 个新稳态）重训的 v3 没有可靠提升：边界的不确定性来自多稳态（新点中 11/48、17/48 两初值类型不同），
  加密网格留一验证基本不变、测试集反而下降，因此最终网络为 v2。复算：`python code/scripts/active_learning.py validate`。
- **球面斑点布局对照（`evidence/sphere_spots_layout/`，负结果）**：把斑点中心换成 45 点互斥能量最小（Thomson）布局，23 个种子中 11 个达到完整稳态且无正模态，不如 CVT 布局（14/23），未采用。
- **球面条纹的稳健性（`evidence/sphere_bandcount/`）**：归档审计中的正模态全部是随机相位选到的同一种条带排布。
  AI 提出相邻条带数候选（6、7，对称纬向条带），每个都收敛到完整稳态后用谱检查择稳：10 个随机轴向 10/10 得到稳定稳态
  （直接 LR 复核一致）。复算：`python code/scripts/sphere_bandcount.py --seeds 1 2 3 4 5 6 7 8 9 10 --workers 8`。
- **网络留一法类型准确率**（青蛙 64%、球面 89%）是分类指标，分母均为旧 9×5 网格的 45 个留一参数点，不等于最终斑图成功率；
  新旧模型的两组留一法网格密度不同，不能直接比。
- **多种子稳健性**（归档审计协议，见 `evidence/EVALUATION_PROTOCOL.md` 与报告）：
  青蛙斑点 31/33；青蛙条纹收敛 33/33 但目标条纹形态 31/33；球面斑点 17/23 完整稳态、14/23 无正模态；
  球面条纹 22/23 完整稳态、9/23 无正模态。这些是历史多种子实验，不是对单个提交文件的验收。
- 所有 AI 输出都经过 PDE 复算；没有使用外部数据或预训练生成模型直接出结果。

## 6. 未包含内容及依据

| 排除项 | 依据 |
| --- | --- |
| `results/ai_generation_fine/`（加密数据新模型） | 未用于正式结果；体积大（模型 3.4 MB×2、数据 1.4 MB） |
| 训练/评测日志（`run.log`、`stdout_*.log`、`robust_*.log`） | 过程日志冗余，且含机器名与运行环境细节 |
| 旧实验、失效候选（六条带 seed0、L5 早期结果、`baseline/*.ply` 等） | 已被正式候选取代或与结论无关 |
| `results/i04_evaluation/` 全量记录（2.6 MB） | 本包不含逐条 I04 记录；精简 main 也不含该全量记录，仅为归档审计摘要 |
| `docs/` 历史版本报告与大纲 | 非本轮提交物；报告按赛题单独上传 |
| `.venv/`、`__pycache__`、`dist/`、`.git/` | 环境/缓存/版本库，不属于答案文件 |
| 本机绝对路径、密钥、个人配置 | 包内 JSON 中仅作说明的仓库路径已改写为相对路径；所有哈希绑定字段（MAT/证书/清单）保持原值，不含任何本机路径、令牌或个人配置 |
| 任务 4 力学调控 | 赛题规定初赛不考核（总决赛内容） |

## 7. 完整性

`PACKAGE_MANIFEST.json` 列出每个文件的包内路径、来源、源 SHA256、包内 SHA256、字节数、保留理由与是否做过路径改写；
仓库根目录命令 `python scripts/i03_verify_package.py --zip <zip> --out <新验收目录>` 会在仓库外独立解包并复核 CRC、清单、MAT 结构/精度/
完整残差、证书哈希绑定、反演一致性、代码与路径卫生。逐文件校验值以该清单为准。

## 当前报告

当前两页报告是 `report/初赛技术报告_v3.pdf`（`python scripts/report_v3.py --font <CJK 字体> --team <队名> --out <新文件>`
生成，数字全部读自提交记录），在 ZIP 外单独提交。`report/初赛技术报告_交付草稿_v2.pdf` 是上一版；
`report/初赛技术报告.pdf` 保留为历史旧稿，其中六条带、相对平衡态及旧成功率表述已过时，不用于当前交付。

当前仓库草稿 ZIP：`submission/drafts/待填_初赛_AI-MECH-2026-002_result.zip`。
`submission/待填_初赛_AI-MECH-2026-002_result.zip` 是保留原字节的历史旧稿，内部证据较旧，不作为本次验收包。
