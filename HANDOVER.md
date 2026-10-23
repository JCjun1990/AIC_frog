# 交接说明（2026-10-23）

基准：队友提交 `c89c2ca`（fix: align submission evidence and reproduction workflow）。本次在其上按评分标准改进，
**正式提交的 4 组生成结果（MAT 与证书）和反演参数 F、k 均未改动**。所有新结果都另放在新目录，并经 Windows 完整验收。

## 1. 结果对比

| 评分项 | c89c2ca | 现在 | 证据 |
| --- | --- | --- | --- |
| 任务 3 生成斑图与目标的相似度 | 随机初值的 21 斑，特征距离 J 1.95，谱峰波长 −10% | AI 多初值选出的 16 斑（目标 15），J **1.55**，波长 −8%，交叉点密度与目标一致；扩大到 246 个候选复核仍最优 | `submission/task3_inversion/best_match/`、`evidence/best_match_search/` |
| 泛化误差（独立测试集 128 个新稳态） | 只有留一法 64% / 89% | 正式网络 v1：69% / 89%；加密数据网络 **v2：92% / 95%**，κ₀ 误差 3.4%→2.7%、2.2%→1.7% | `evidence/generalization/` |
| 端到端效率（青蛙 8 留出点 × 3 种子） | AI v1 比纯 PDE 少约 1.7 倍步数 | AI v2 约 **2.4 倍**（10 875 对 25 875 步，23/24 完成） | `evidence/active_learning/` |
| 球面条纹稳健性 | 9/23 无正模态 | AI 提出 6、7 条带候选、谱检查择稳：**10/10** 轴向稳定（直接 LR 复核一致） | `evidence/sphere_bandcount/` |
| 校验可移植 | `verify` 只在 Windows x86 上通过 | 其他 CPU 上若只差算子末位舍入，改为核对网格哈希、总质量并重算残差；篡改仍会被拒 | `tests/test_portable_verify.py` |
| 命名打包 | 填队名后包里仍写"草稿、不得上传"，验收判失败 | 去掉草稿段落、不打入内部 VALIDATION.md；测试队名打包验收 PASS | `scripts/i03_package.py` |
| 报告 | v2（内部代号多） | **v3**：按评分细则逐项写，数字全部从记录读取 | `submission/report/初赛技术报告_v3.pdf` |
| 校核 | 89 项测试 | 95 项测试；Windows 打包验收 22 项 PASS | `DELIVERY_VALIDATION.json` |

试过但**没有采用**的（证据里如实保留）：物理特征输入 v2p（无提升）；类型边界主动学习 v3（边界不确定性来自多稳态，
验证无提升、测试集下降）；球面斑点 Thomson 最小能量布局（11/23，不如 CVT 的 14/23）；另加 180 个任务 3 候选（未超过 1.554）。

## 2. 改动清单

新增脚本（都已加入提交包 `code/scripts/`）：

| 脚本 | 作用 |
| --- | --- |
| `scripts/inversion_best_match.py` | 任务 3：AI 多初值 + Newton，选最接近目标的稳定稳态；`--install` 装入提交包 |
| `scripts/report_v3.py` | 两页报告 v3（PyMuPDF + CJK TrueType 字体；`--team` 填队名） |
| `scripts/generalization_eval.py` | 独立测试集上的泛化误差（v1 / v2 / v2p） |
| `scripts/active_learning.py` | 主动学习选点、标注、验证；端到端效率对比 |
| `scripts/sphere_bandcount.py` | 球面条纹条带数候选 + 谱检查择稳 |
| `scripts/sphere_spots_layout.py` | 球面斑点 Thomson 布局对照（负结果） |

修改：`scripts/i03_package.py`（命名打包、新脚本列表）、`scripts/i03_verify_package.py`（检查最佳匹配稳态；命名包检查 named 状态）、
`scripts/submission_workflow.py`（可移植校验）、`submission/README_提交.md`、`README.md`、`DELIVERY_VALIDATION.json`、`.gitignore`。

提交包（`submission/`）：新增 `task3_inversion/best_match/` 和 `evidence/` 下 5 个目录（generalization、sphere_bandcount、
active_learning、best_match_search、sphere_spots_layout）；`PACKAGE_MANIFEST.json`、`code/scripts/` 与 `drafts/` 草稿 zip 已按
Windows 重建结果同步（zip SHA256 `465b629a…`）。

`results/` 下是各次运行的摘要与候选记录；原始中间文件（含机器名的原始数据、计算记录、日志、未采用的模型）已在 `.gitignore` 中排除。
`results/ai_generation_fine/*/model.joblib` 是 v2 网络（报告里的泛化数字就来自这两个文件）。

## 3. 复现命令

| 内容 | 命令 | 参考用时 |
| --- | --- | --- |
| 单元测试 | `python -m unittest discover -s tests` | 秒级 |
| 任务 3 最佳匹配 | `python scripts/inversion_best_match.py --workers 20` | Windows 20 进程约 38 分钟 |
| 泛化误差 | `python scripts/generalization_eval.py testset --workers 22`，再 `train`、`evaluate` | 测试集约 70 分钟（Windows） |
| 主动学习 / 端到端 | `python scripts/active_learning.py select / run / train / validate / e2e --model v2` | 各约 20–60 分钟 |
| 球面条纹择稳 | `python scripts/sphere_bandcount.py --seeds 1 2 3 4 5 6 7 8 9 10 --workers 8` | Mac 约 25 分钟 |
| 报告 | `python scripts/report_v3.py --out <新文件> --font <CJK 字体> --team <队名>` | 秒级（Mac 用 `/Library/Fonts/Arial Unicode.ttf`，Windows 用 `C:\Windows\Fonts\simhei.ttf`） |
| 打包与验收 | `python scripts/i03_package.py --team <队名> --out <目录>`；`python scripts/i03_verify_package.py --zip <zip> --out <目录> --report <pdf>` | 约 5 分钟 |

## 4. 注意事项

- **队名仍是"待填"**。上传前：`i03_package.py --team 队名` 重新打包，`report_v3.py --team 队名` 重新生成报告，再跑一次验收。
- `submission/` 根目录的旧 zip 和 `report/初赛技术报告.pdf`、`_交付草稿_v2.pdf` 都是旧稿，**上传时别拿错**（建议删除，历史在 Git 中）。
- 仓库目前是公开的；推送后所有新结果对外可见。
- 正式 4 组结果仍由 v1 网络生成，报告与 README_提交 已写明"复赛更新改用 v2"。
- Windows 笔记本上的工作目录是 `C:\Users\18650\turing2`（精简 main 的副本）；venv 已装 PyMuPDF。打游戏前确认没有 Python 进程。

## 5. 待办（按优先级）

1. 定队名，正式打包、生成报告、验收（约 10 分钟）。
2. 队友审阅报告 v3。
3. 可选：球面斑点多候选择稳，把稳健性 14/23 提上去（约 2–4 小时）。
4. 可选：用 v2 重新生成正式结果并重出证书（约 3–4 小时，主要是一致性，对分数影响小）。
5. k 偏低 3.2%：没有确定的修法，真值已知时改方法易被视为调参，暂不建议。
6. 复赛报告（≥1000 字）；总决赛任务 4（力学调控）与 PPT（≤20 页）。
