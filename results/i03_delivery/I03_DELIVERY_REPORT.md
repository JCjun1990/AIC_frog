# I03 交付报告：初赛技术报告、正式结果与提交包

- 仓库：`D:\Code\AIC_frog`，分支 `codex/i01-float64-mat`，HEAD `065396591aef328c3484cb47e196cf5f8be71b6c`
- 本轮产物目录：`results/i03_delivery/`（未覆盖 I01/I02/I04 任何文件；受保护的 1096 个文件哈希复核 0 变更）
- 交付结论：**工程交付验收通过（草稿状态）**。四组正式 MAT、反演结果、两页报告与 72 文件提交包均已生成，包在仓库外独立解包后 21/21 项检查通过。
  **但队名尚未提供，ZIP 与报告均为 `待填` 占位，不能直接上传。** 本轮不代表组委会已接收或保证得分。

---

## A. 官方提交要求与逐项符合情况

依据：根目录 `附件3.生物曲面扩散动力学及图灵斑图生成、演化与调控赛题规则.pdf`（4 页；公式与参数在第 1–2 页为矢量图，已逐页渲染核对）。

| 官方要求 | 规则位置 | 本包状态 | 证据 |
| --- | --- | --- | --- |
| 初赛考核任务 1（单位球面生成）、任务 2（青蛙曲面生成）、任务 3（反演）；任务 4 只在总决赛 | 二（四类任务）、四.3 | **完成** | `task1_sphere/`、`task2_frog/`、`task3_inversion/` |
| 所有方案须含 AI；鼓励 AI 与曲面 PDE 求解器结合 | 三.1 | **满足** | MLP 初值 + GP 反演代理；所有 AI 输出经 PDE 复算；报告 §2 |
| 只用附件 `frog.mat`、`Solution.mat`；不得用外部标注数据或预训练生成模型直接出结果 | 三.2 | **满足** | 包内 `code/` 附原件；README §5；训练标签由求解器自生成 |
| 答案文件按 `teamName_round_problemID_result` 命名，ZIP，根目录不嵌套无关目录 | 六.1 | **符合（占位名）** | `待填_初赛_AI-MECH-2026-002_result.zip`，内容在根目录 |
| 结果须含：斑图可视化、可复现代码、PDE 残差/稳定性/收敛记录 | 六.1(1)(2)(3) | **完成** | 每任务 `final_v.png`、`summary.json`、`stability_certificate.json`、`code/` |
| 反演任务须含反演参数、目标与生成对比、误差指标 | 六.1(4) | **完成** | `task3_inversion/summary.md`（参数表、误差、特征对照）、`final_vs_target.png` |
| 力学调控方案/前后对比/稳定性 | 六.1(5) | **不适用** | 初赛不考核，总决赛任务 4 |
| 技术报告初赛 ≤ 2 页，PDF 单独在“文档类文件”栏目提交 | 六.2、提交要求 | **完成（2 页）** | `report/初赛技术报告_I03_草稿.pdf`，单独上传，不放入 ZIP |
| 报告须说明关键公式与推导、AI-PDE 架构、数据使用、参数依据、单位量纲、稳定性、校核结论、资源控制 | 六.2 | **覆盖** | 报告 §1–§7 |
| 截止时间、提交次数、文件大小/格式以当轮通知为准 | 六 | **未知** | 规则 PDF 未给出具体数值，需以官网/QQ 群通知为准 |
| 答辩材料（总决赛 PPT ≤ 20 页） | 六.3 | **不适用** | 本轮不涉及 |

初赛实际交付两项：**答案 ZIP**（本题结果文件）与**技术报告 PDF**（单独文档类文件）。本轮没有生成答辩材料。

## B. 四组正式候选（按哈希自动定位并绑定）

选择链：I04 评测记录 `experiment_id` → MAT 文件 → SHA256 同时等于 I04 记录与 I01 证书 → 用实际 robust 算子重算完整残差 → I02/I03 稳定性证书的 `input.mat_sha256` 必须等于同一哈希。清单：`results/i03_delivery/candidates.json`。

| 任务 | 文件 | SHA256 | 字节 | 完整残差 | 形态 | 谱状态 / max Re λ | 证书 |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| 任务 1 球面斑点 | `results/i01_float64_verified/tasks/sphere6_spots/solution.mat` | `44f5931a623d15600dd5811f8d9366c049d732b917c7c1691f44a65f5c0586f3` | 655704 | 8.422603e-15 | spots / 45 | numerical_check_passed / −6.231004e-06 | `results/i02_stability/certificates/spots/stability_certificate.json` |
| 任务 1 球面条纹 | `results/i02_stability/candidates/seed2_v2/solution.mat` | `ef132d82c549a78912be21e3ef70edb31725e7e5a74d1793a0e73c15c0dcae10` | 655704 | 2.436627e-14 | stripes_or_maze / 7 | numerical_check_passed / −1.477798e-06 | `results/i02_stability/certificates/seed2/stability_certificate.json` |
| 任务 2 青蛙斑点 | `results/i01_float64_verified/tasks/frog_spots/solution.mat` | `19a92c65ec759c9a7fa4e58ceaf9d5b9716e9eea660a23f1e87651f11bd737cb` | 464472 | 2.171130e-14 | spots / 22 | numerical_check_passed / −1.034897e-04 | `results/i03_delivery/frog_stability_v2/frog_spots/frog_stability_certificate.json` |
| 任务 2 青蛙条纹 | `results/i01_float64_verified/tasks/frog_stripes/solution.mat` | `aa99f1373d76cc704c62ea87c8daabd7ef74adf0627c6d24eca3aa25b0150a82` | 464472 | 1.005659e-14 | stripes_or_maze / 5 | numerical_check_passed / −1.548304e-04 | `results/i03_delivery/frog_stability_v2/frog_stripes/frog_stability_certificate.json` |

- 残差定义：`sqrt((RMS_M(Ru)^2 + RMS_M(Rv)^2)/2)/F`，实际 robust L/M，**不作转动投影**。
- 四份 MAT 均为 `Solution.u/v`、`(n,1)`、float64，数值有限，`u∈[0,1]`、`v≥0`；结构/精度/范围/残差四项全部通过。
- 历史六条带 seed0（`1a903c83…`）因 +1.6654e-06 的可靠正倾斜模态，保留为历史记录，不作为正式结果（`candidates.json.historical_only`）。

## C. 青蛙两组新增谱检查（I03 核心补充）

脚本 `scripts/i03_frog_stability.py`，输出 `results/i03_delivery/frog_stability_v2/`。复用 I02 的 Jacobian、质量相似变换、ARPACK、特征对残差与保守判定，但**完全不做转动标注/投影**（青蛙曲面无连续旋转对称性）。

| 任务 | 配置 | 收敛 | 采样数 | max Re λ | 最大特征对残差 |
| --- | --- | --- | --- | ---: | ---: |
| 青蛙斑点 | σ = 1e-3，k = 12 | 是 | 12/12 | −1.034897401906e-04 | 4.416e-15 |
| 青蛙斑点 | σ = 0，k = 20 | 是 | 20/20 | −1.034897401906e-04 | 1.504e-14 |
| 青蛙斑点 | σ = 1e-5，k = 8 | 是 | 8/8 | −1.034897401906e-04 | 3.556e-15 |
| 青蛙条纹 | σ = 1e-3，k = 12 | 是 | 12/12 | −1.548304287718e-04 | 4.884e-15 |
| 青蛙条纹 | σ = 0，k = 20 | 是 | 20/20 | −1.548304287718e-04 | 5.020e-15 |
| 青蛙条纹 | σ = 1e-5，k = 8 | 是 | 8/8 | −1.548304287718e-04 | 3.814e-15 |

- 三配置最大值一致（差 < 1e-14），两组共 80 个特征对残差 ≤ 1.5e-14；`decision.status = numerical_check_passed`，`sampled_signs_resolved_negative = true`。
- **无移位 `which='LR'` 搜索失败并保留记录**：斑点 3001 次迭代仅 2/8 收敛（234.7 s，部分结果 max −8.258e-4）；条纹 0/8（263.4 s）。该失败归因于刚性算子（谱半径远大于右端特征值间隙），记录在证书 `unshifted_lr_attempt`，**不用于通过判定**；通过判定改用第三个正侧 shift-invert 配置（σ=1e-5，位于采样最大值之上，任何正模态都会比最右负模态更靠近该移位）。
- 与 I01 之前旧场的单 shift 记录一致：斑点相对差 2.3e-12，条纹 3.6e-11，说明新检查与历史诊断相互印证。
- 证书同时保留 `reference_i02_rule`：I02 规则要求直接 LR，故对同一批报告只能给出 `stability_not_closed`；两种判定并列，差异原因写在证书里。
- 局限：k=8–20 对 58020 维算子的有限采样，不是全谱证明；无时域扰动测试；不作全局非线性稳定性结论。

## D. 初赛技术报告（I03 修订）

- 原报告只有 PDF（`docs/初赛技术报告.pdf` 2 页、`_2026-10-18.pdf` 旧版），**仓库中没有可编辑源文件**；原 PDF 未被修改（`git status docs` 干净）。
- 本轮新建可编辑源并导出 PDF：
  - 源：`results/i03_delivery/report/初赛技术报告_I03_草稿.docx`（python-docx 生成，420 KB）
  - 成品：`results/i03_delivery/report/初赛技术报告_I03_草稿.pdf`（LibreOffice 转换，**2 页**，401 KB）
  - 数据：`results/i03_delivery/report/report_data.json`（报告中每个数字的读取来源）
- 结构：摘要 → §1 模型/数据/量纲 → §2 离散与 AI-PDE 架构 → §3 正式生成结果（含表 1 与四联图）→ §4 反演（含参数表）→ §5 稳健性/稳定性边界/局限 → §6 校核、资源与交付 → §7 结论。
- 所有结果数字来自 `candidates.json`、I03 青蛙证书、`results/inversion/summary.json`、I04 `run_manifest.json`、`cv.json` 与任务记录，无手工抄写；图 1 由四份提交 MAT 直接渲染。
- 相对旧报告的关键修订：
  1. 球面条纹由“六条带、排除倾斜模后 max −4.6e-5”改为“七条带 seed2、完整残差 2.437e-14、三配置 max −1.478e-06”，并注明六条带因 +1.67e-06 正模态停用；
  2. 球面斑点残差由 4.3e-12 改为 I01 MAT 实测 8.423e-15，max Re λ 由 −5.9e-5 改为 −6.231e-06；
  3. 青蛙两组残差改用提交 MAT 的 2.171e-14 / 1.006e-14，并新增 I03 三配置谱检查；
  4. 稳健性改为统一口径：青蛙条纹 33/33 收敛但条纹形态 31/33；球面斑点 17/23 → 14/23 无正模态；球面条纹 22/23 → 9/23；
  5. 反演补充“k 真值落在不确定带外 0.0007”与波长/缺陷未完全匹配的说明；
  6. 明确“有限采样不是全谱证明”“L5 只有相对平衡态”“墙钟加速比仅在旧模型同负载一组成立”。
- 草稿标记：标题下红色行“交付草稿：队名待填写…”、副标题“队伍：待填”、页脚含赛题号与页码；**未标记为可直接上传**。
- 页面控制：A4、上下 1.45 cm、左右 1.6 cm、正文 9.4 pt；导出后 2 页（规则上限 2 页）。
- 报告文件名规则赛题未规定（只要求 PDF）；建议上传前改为 `<队名>_初赛_AI-MECH-2026-002_报告.pdf`。
- 项目测试：`python -m unittest discover -s tests` 共 **83 项全部通过**（日志 `results/i03_delivery/tests.log`），I01/I02/I04 无回归。

## E. 提交包（`package_v2`）

- 包目录：`results/i03_delivery/package_v2/待填_初赛_AI-MECH-2026-002_result/`
- ZIP：`results/i03_delivery/package_v2/待填_初赛_AI-MECH-2026-002_result.zip`（**72 文件，11.1 MB**；解压 13.1 MB）
- 结构（71 条清单 + 清单本身）：

```
README_提交.md                    提交说明（结果、反演、复现、口径、排除项）
PACKAGE_MANIFEST.json             71 个文件的来源、源 SHA256、包内 SHA256、字节、保留理由
task1_sphere/ (11)                spots/ stripes/ 各 4 文件；ai_model/ 3 文件
task2_frog/ (11)                  同上
task3_inversion/ (10)             summary.md/json、solution.mat、4 张图、3 个 CSV
evidence/ (8)                     AI vs 纯积分汇总、96 次评测 CSV、基线汇总、稳定性横向比较、4 个校核文件
code/ (30)                        turing_rd 16、scripts 11、requirements.txt、frog.mat、Solution.mat
```

- 保留理由（摘要，逐文件见 `PACKAGE_MANIFEST.json`）：
  - 四组 `solution.mat` + `summary.json` + `stability_certificate.json` + `final_v.png`：规则六.1(1)–(3) 的可视化、残差/稳定性记录与正式结果；
  - `ai_model/{model.joblib,dataset.json,cv.json}`：复现 AI 初值与留一法记录所必需（模型是本方案 AI 部分）；
  - `task3_inversion/*`：规则六.1(4) 的参数、目标—生成对比与误差指标；
  - `evidence/*`：AI-PDE 融合程度、泛化与效率、求解器/几何校核的可核查记录；
  - `code/*`：求解器、AI、反演、稳定性与评测脚本 + 两个组委会附件（复现依赖）。
- 路径卫生：包内 JSON 中 5 个文件的**说明性绝对路径**已改写为相对路径（`task1_sphere/spots/summary.json`、`task1_sphere/spots/stability_certificate.json`、`task1_sphere/stripes/stability_certificate.json`、`task2_frog/{spots,stripes}/summary.json`）；所有哈希绑定字段保持原值，清单同时记录源 SHA256 与包内 SHA256。
- 未包含内容与依据见 `README_提交.md` §6（新模型 `ai_generation_fine`、日志、失效候选、全量 I04 记录、历史报告、环境/缓存、任务 4 等）。

## F. 独立解包验收

- 方法：`scripts/i03_verify_package.py` 把 ZIP 解压到**仓库外**的全新临时目录；只使用解压后的字节，导入**包内** `code/turing_rd`、用包内 `frog.mat` 重新构造几何并重算残差；结束后删除临时目录，报告留在 `results/i03_delivery/acceptance_v2/`。
- 第一次构建（`package/`）验收**失败**，原因是 2 项：README 未进清单、5 个 JSON 仍含本机绝对路径（转义形式未被初版清洗器识别）。已修正（清洗器改为解析后逐字符串替换；README 进入清单），旧包与旧验收记录保留在 `package/`、`acceptance/`，作为迭代证据。
- 第二次构建（`package_v2/`）验收 **PASS，21/21**：

| # | 检查 | 结果 | 关键证据 |
| --- | --- | --- | --- |
| 1 | ZIP CRC | PASS | 72 项，无坏块 |
| 2 | 根目录布局 | PASS | README 与清单在根目录，无外壳目录 |
| 3 | 相对 POSIX 路径 | PASS | 无绝对路径/`..`/反斜杠 |
| 4 | 必需交付物齐全 | PASS | 34 项必需文件全部存在 |
| 5–8 | 清单完整/哈希/全列出/理由来源 | PASS | 71 条；包内 SHA256 全部匹配；无未列出文件 |
| 9–11 | 无本机路径 / 无密钥 / 无任务 4 | PASS | 0 offender |
| 12 | 包内代码可导入 | PASS | `turing_rd` 从解压目录加载 |
| 13 | 四份 MAT 验收 | PASS | 重算残差与记录值完全一致；哈希/证书绑定一致；三配置齐全 |
| 14–17 | 反演参数/MAT/误差/图 | PASS | F=0.027712，k=0.060040；反演 MAT 残差 9.061e-14，21 斑；4 张 PNG 有效 |
| 18–19 | 报告页数与数字一致 | PASS | 2 页；8 个残差/特征值字符串全部命中 |
| 20–21 | 命名与草稿状态 | PASS | `待填_初赛_AI-MECH-2026-002_result.zip`；manifest 标注 draft |

- 未做的验证：没有从头重训模型，也没有重跑 20 万时间单位的生成或反演全过程（验收要求允许轻量复现）；因此验收证明“结果、证书、代码与清单自洽可复现”，不证明重新训练会逐位复现（模型文件与训练数据已随包提供）。

## G. 缺失信息与未解决风险

1. **队名缺失（唯一上传阻塞项）**：ZIP 名、报告副标题、页脚作者均为 `待填`。补齐后需重新生成包与报告。
2. 官方**截止时间、提交次数、ZIP 大小上限、报告模板/封面**未在附件 3 中给出，需以官网或 QQ 群通知为准；本包按“≤2 页 + 根目录不嵌套”执行。
3. 青蛙谱检查的**证据等级**低于球面：三配置均为 shift-invert；直接 LR 不收敛已记录但不作为通过证据；仍属有限采样，不是全谱证明。
4. 反演：k 的题给值落在不确定带外 0.0007；特征波长 −10%、κ_v +20%、缺陷比例 0.25 对 0.60，**若干尺度与缺陷特征未完全匹配**；目标数据本身带噪声/负 v 区。
5. L5 评测没有 orient，只有相对平衡态；多种子稳健性数字来自历史协议，不对应单个提交文件。
6. 报告为新建文档（原报告无可编辑源），版式与旧 PDF 不完全相同；旧 PDF 保留未动。
7. 提交包不包含报告 PDF（按赛题分开提交）与任务 4 内容（初赛不考核）。

## H. 可直接提交的文件与仍受阻的部分

- **可上传（补队名后）**：
  - 答案 ZIP：`results/i03_delivery/package_v2/待填_初赛_AI-MECH-2026-002_result.zip`
  - 技术报告 PDF：`results/i03_delivery/report/初赛技术报告_I03_草稿.pdf`
- **不能直接上传**：上述两份文件的队名、报告作者/队伍信息均为 `待填`；ZIP 文件名不符合真实队名规范。
- 补齐队名后的最终命令（新目录，不覆盖本次草稿）：

```bash
python scripts/i03_report_docx.py                 # 修改脚本顶部 TEAM 后重建 DOCX
<libreoffice-kit> convert --input ...docx --output ...pdf
python scripts/i03_package.py --team <队名> --out results/i03_delivery/package_final
python scripts/i03_verify_package.py \
  --zip results/i03_delivery/package_final/<队名>_初赛_AI-MECH-2026-002_result.zip \
  --out results/i03_delivery/acceptance_final \
  --report results/i03_delivery/report/<新报告>.pdf
```

## I. 后续轮次（不属于本轮）

- **复赛**（与初赛合并考核，可更新初赛结果）：更新结果后重新走 I01–I04 与打包/验收流程；复赛报告 ≥1000 字；包名中的轮次改为复赛。
- **总决赛**：任务 4（曲率/生长变形调控）尚未实施；答辩 PPT ≤20 页，需补充方法创新、效率与仿生设计。
- **技术积压**：L5 评测补 orient；青蛙稳健性种子（13/16/21 与六条带代表态）补 I02 式审计；六条带正倾斜模的网格加密/朝向对照；孔洞区形态先验；反演波长/缺陷特征的进一步匹配。

---

### 本轮新增/修改文件

| 文件 | 作用 |
| --- | --- |
| `scripts/i03_frog_stability.py` | 青蛙两组独立谱检查与证书（v2 三配置；v1 保留直接 LR 失败记录） |
| `scripts/i03_candidates.py` | 按实验 id + SHA256 自动定位四组正式候选并绑定证书 |
| `scripts/i03_render_figures.py` | 从提交 MAT 渲染单任务图与报告四联图 |
| `scripts/i03_report_docx.py` | 从验证记录生成两页报告 DOCX 与 `report_data.json` |
| `scripts/i03_package.py` | 最小完整提交包、逐文件清单与 ZIP（含路径清洗） |
| `scripts/i03_verify_package.py` | 仓库外独立解包验收（21 项检查） |
| `results/i03_delivery/` | 候选人清单、青蛙证书 v1/v2、图、报告、包 v1/v2、验收 v1/v2 |

`git status` 显示本轮只新增未跟踪文件；4 个 I01 tracked 修改保持原样；未 commit、未 push、未 reset/clean。
