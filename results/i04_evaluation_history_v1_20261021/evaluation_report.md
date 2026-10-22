# I04 统一评测体系与结果重统计

- evaluation_version: `aic-frog-eval-v1`
- generated: 2026-10-21T13:01:24.868363+00:00
- git: `065396591aef328c3484cb47e196cf5f8be71b6c` on `codex/i01-float64-mat` (dirty=True; I01/I02 changes stay uncommitted by design)
- code: `scripts/i04_evaluate.py`=a24df17a79f7, `scripts/json_safe.py`=a199b039bdc5, `turing_rd/evaluation.py`=50575ac68407
- 本文件由 `scripts/i04_evaluate.py` 从原始 JSON/PKL、I01 MAT 证书和 I02 稳定性证书机械生成；所有数字都可回溯到 `evaluation_records.json` 的记录与 `run_manifest.json` 的源文件哈希。

## 1. 判据定义（不同数学概念分开记录）

| 判据 | 定义 | 阈值 | 适用范围 | 证据来源 |
|---|---|---|---|---|
| `file_validity` | Solution.u/v read back as float64 with the required (n,1) layout, all values finite, u in [0,1], v >= 0, no float32-quantized nonuniform field, and any attached stability certificate bound to the SHA256 of the same MAT bytes. | structure+precision+range+residual certificate accepted; certificate mat_sha256 == file sha256 | only runs whose field was saved (formal candidates); robustness/evaluation scans saved no field | turing_rd/result_io.py verify_mat / validate_certificate; results/i02_stability certificates |
| `steady_full` | Complete stationary PDE residual sqrt((RMS_M(Ru)^2+RMS_M(Rv)^2)/2)/F computed from the saved/recorded field with the actual robust L and M, with NO rotation projection. | <= 1e-11 | every run that recorded a full residual | turing_rd/diagnostics.full_steady_residual; I01 certificate residual; record rel_residual |
| `relative_equilibrium` | Residual after removing the best rigid-rotation fit is below tolerance while the full residual is not: the state is steady only up to a rigid rotation. | rotation-removed <= 1e-11 AND full > 1e-11 | spherical runs that recorded rel_residual_rot; not applicable to the frog mesh | turing_rd/steady.rotation_split; record rel_residual_rot |
| `spectral` | Numerical linear stability of the recorded state from a finite eigenvalue sample. Rotation/tilt-labelled eigenvalues are NEVER excluded from a pass decision. | pass iff every sampled |Re lambda| > 1e-09 and no sampled Re lambda > 0; any Re lambda > 0 -> unstable_positive_mode; |Re lambda| <= floor -> insufficient | only runs where eigenvalues were sampled; otherwise not_checked | turing_rd/stability.py (I02 certificates) or legacy single-shift records |
| `shape` | Observed morphology equals the task/reference morphology for that parameter point. | type == 'spots' for spot tasks; type == 'stripes_or_maze' for stripe tasks; band count is reported separately | every run with a recorded pattern type | turing_rd/diagnostics.pattern_stats; record type / predicted_type |
| `budget` | The run finished inside its declared physical-time cap instead of stopping at the cap. | relax_time < t_max (evaluation/robustness t_max = 2e5; training labels t_max = 1e5) | every run with a recorded relax_time | record relax_time; scripts/ai_generation.py schedules |
| `end_to_end` | Task success: steady_full AND spectral pass AND shape match AND within budget, with an invalid saved field vetoing the result. A relative equilibrium alone is not a success. | all of the above | every record | this module, task_success() |

谱状态四种取值：`numerical_check_passed`、`unstable_positive_mode`、`insufficient_evidence`、`not_checked`。A finite ARPACK sample with k << dim(J) is numerical evidence only; it is never a proof that the whole spectrum lies in the left half plane.

## 2. 四组任务的新统计（原始记录机械重算）

| 任务 | 计划 | 执行 | 完整稳态 | 仅相对平衡 | 谱已检查 | 无正模态 | 有正模态 | 谱未检查 | 形态匹配 | 预算内 | 旧口径 | **新口径** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frog_spots | 33 | 33 | 32 | 0 | 32 | 31 | 1 | 1 | 33 | 32 | 31/33 | **31/33** |
| frog_stripes | 33 | 33 | 33 | 0 | 33 | 33 | 0 | 0 | 31 | 33 | 33/33 | **31/33** |
| sphere6_spots | 23 | 23 | 17 | 0 | 17 | 14 | 3 | 6 | 23 | 17 | 17/23 | **14/23** |
| sphere6_stripes | 23 | 23 | 22 | 0 | 22 | 9 | 13 | 1 | 23 | 22 | 22/23 | **9/23** |

- **frog_spots**：正模态种子 [22]；预算耗尽 [25]；形态不符 无；稳态未达标 [25]；n_components 分布 {'22': 10, '21': 8, '23': 8, '20': 1, '19': 1, '24': 3, '25': 1, '18': 1}。
- **frog_stripes**：正模态种子 无；预算耗尽 无；形态不符 [1, 21]；稳态未达标 无；n_components 分布 {'5': 12, '9': 1, '6': 8, '8': 1, '7': 7, '3': 2, '10': 1, '4': 1}。
- **sphere6_spots**：正模态种子 [13, 16, 21]；预算耗尽 [6, 7, 15, 17, 18, 20]；形态不符 无；稳态未达标 [6, 7, 15, 17, 18, 20]；n_components 分布 {'45': 23}。
- **sphere6_stripes**：正模态种子 [0, 4, 5, 6, 7, 9, 10, 13, 14, 15, 16, 17, 21]；预算耗尽 [1]；形态不符 无；稳态未达标 [1]；n_components 分布 {'6': 13, '7': 10}。

说明：旧口径 = 决定性残差 ≤ 1e-11 且 legacy stable 为真；旧 stable 把旋转/倾斜模态排除在 max Re λ 之外，并容忍 max Re λ < 1e-6。新口径 = 完整稳态 + 采样谱无正实部（含旋转类模态）+ 形态匹配 + 预算内。两者的差异来自谱口径（是否排除旋转类模态）与形态口径，不是重新求解。

## 3. 旧统计与新统计的差异

| 任务 | 旧发布 | I04 重算旧口径 | I04 新口径 | 主要差异来源 |
|---|---|---|---|---|
| frog_spots | 31/33 | 31/33 | **31/33** | 2 个失败：seed22 收敛但 max Re λ=+3.2e-5（鞍点）；seed25 到 t=2e5 未收敛（3.9e-3） |
| frog_stripes | 33/33 | 33/33 | **31/33** | 收敛 33/33、谱全负，但 seed1（9 斑）与 seed21（10 斑）最终是斑点态 → 条纹形态 31/33 |
| sphere6_spots | 17/23 | 17/23 | **14/23** | 6 个预算失败（seed 6/7/15/17/18/20）；17 个完整稳态中 seed13/16/21 有正旋转模态 → 14/23 |
| sphere6_stripes | 22/23 | 22/23 | **9/23** | 1 个预算失败（seed1）；22 个完整稳态中 13 个六条带有正倾斜模态 +1.67e-6 → 9/23 |

- `robustness.json` 的 `max_re_lambda` 数组只收录 legacy 成功运行（31/17/22 项），因此旧文里的 “max Re λ 范围” 是**条件于成功**的统计，不是全部样本的稳定性分布。
- 旧发布把“收敛到任意稳定态”当成任务成功：青蛙条纹 33/33 里含 2 个斑点态，球面条纹 22/23 里含 13 个正倾斜模态。
- 球面斑点 17/23 里含 3 个正旋转模态；14/23 才是“完整稳态且未发现正模态”的候选数。

### 旧表述逐条核对

| 旧表述 | 数值 | 来源 | I04 判定 | 说明 |
|---|---|---|---|---|
| 青蛙斑点稳健性 31/33 | 31/33 | README.md:755; results/ai_generation/tasks/robustness.md:8 | **数字保留，口径需补充** | 31 = 完整稳态且未发现正模态；失败：seed22 收敛但 max Re λ=+3.2e-5（鞍点），seed25 预算耗尽（3.9e-3） |
| 青蛙条纹稳健性 33/33（含 2 个斑点态） | 33/33 | README.md:756; robustness.md:9 | **数字保留，不能称为目标条纹成功率** | 33/33 只是“收敛且谱负”；条纹形态 31/33（seed1 为 9 斑、seed21 为 10 斑） |
| 球面斑点稳健性 17/23 | 17/23 | README.md:757; robustness.md:10 | **拆分为多个指标** | 17/23 完整稳态；seed13/16/21 有正旋转模态 → 14/23 无正模态候选；6 个预算失败 |
| 球面条纹稳健性 22/23 | 22/23 | README.md:758; robustness.md:11 | **拆分为多个指标** | 22/23 完整稳态；13 个六条带有正倾斜模态 +1.67e-6 → 9/23 无正模态；1 个预算失败（seed1） |
| 青蛙 AI 24/24 收敛且稳定、1.7 倍加速 | 24/24, 1.7x | README.md:739-740; scripts/make_submission.py:178 | **24/24 保留；加速比只对旧模型同负载那一组成立** | 24 次都是完整稳态且谱全负（严格口径同样 24/24）；同负载墙钟 solve_s 673/1159 s （total_s 678/1166 s）可比，步数比 15062/25875≈0.58 |
| 青蛙新模型 23/24 | 23/24 | README.md:801 | **保留** | 1 个失败为 task_spots 的 seed2（未收敛） |
| 球面 L5 AI 12/24（旧）/10/24（新） | 12/24, 10/24 | README.md:812 | **必须改写为“相对平衡态 + 旧谱口径”** | 24 次里没有完整稳态；严格口径（相对平衡 + 无正模态）为 10/24、6/24，普通流程 4/24 |
| 球面条纹 seed0（六条带）排除 2 个倾斜模后稳定 | stable after excluding tilt modes | README.md:836,887 | **撤回** | 同一 MAT 上 I02 三配置确认 +1.6654e-6 是可靠正模态（特征对残差 + 扰动增长） |
| 球面斑点为完整稳态且稳定 | stable | README.md:835; 交接说明任务1 | **保留但需限定** | 完整稳态 8.423e-15；I02 三配置无正模态（max Re λ=-6.231e-6）；有限采样，不是全谱证明 |
| 留一法类型准确率 青蛙 64%→86%，球面 89%→93% | 64%->86%, 89%->93% | README.md:786-789 | **数字保留，但不可同难度比较** | 旧 45 点、新 306 点；去掉一点后邻点间距差 2 倍，两者不是同一难度的留一法 |

## 4. 正式候选与证书绑定

| 候选 | MAT sha256 | 完整残差 | 形态 | 分量数 | 谱状态 | 谱 max Re λ | 证据等级 | 文件 | I01 | 端到端 | 推荐 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frog_spots | `19a92c65ec759c9a` | 2.17113e-14 | spots | 22 | not_checked |  | none | valid | true | false | true |
| frog_stripes | `aa99f1373d76cc70` | 1.00566e-14 | stripes_or_maze | 5 | not_checked |  | none | valid | true | false | true |
| sphere6_spots | `44f5931a623d1560` | 8.4226e-15 | spots | 45 | numerical_check_passed | -6.231e-06 | i02_unprojected_three_configuration_k12_k20_k8_with_eigenpair_residuals | valid | true | true | true |
| sphere6_stripes_6band | `1a903c837104ace4` | 3.28881e-15 | stripes_or_maze | 6 | unstable_positive_mode | 1.66543e-06 | i02_unprojected_three_configuration_k12_k20_k8_with_eigenpair_residuals | valid | true | false | false |
| sphere6_stripes_seed2 | `ef132d82c549a789` | 2.43663e-14 | stripes_or_maze | 7 | numerical_check_passed | -1.4778e-06 | i02_unprojected_three_configuration_k12_k20_k8_with_eigenpair_residuals | valid | true | true | true |

- 证书与 MAT 通过 SHA256 绑定：`certificate.input.mat_sha256 == sha256(solution.mat)`；任一不一致脚本直接失败（fail-closed）。
- 青蛙两个候选只有 I01 证书；旧稳定性数字来自 **I01 之前的场**（原 MAT 哈希记录在 I01 provenance），不是对当前 float64 MAT 的谱检查，因此 `spectral_status=not_checked`。
- 六条带 seed0 保留为历史结果：I02 在**同一份 MAT** 上得到可靠正倾斜模态 +1.6654e-6，不能推荐正式提交；七条带 seed2 的证书绑定到新 MAT，谱检查通过。
- seed2 的 I02 证书只覆盖重建后的那一份 MAT；其他 8 个收敛的七条带种子（3/8/11/12/18/19/20/22）没有独立证书，只有旧单 shift 证据，不能用 seed2 的证书代替。

## 5. AI 泛化与效率：原先不公平或不严谨的比较

### 5.1 泛化指标

| 指标 | 曲面 | 旧模型 | 新模型 | 可直接比较 | 备注 |
|---|---|---|---|---|---|
| cv_type_accuracy | frog | 0.644444 | 0.862745 | false | leave-one-grid-point-out with 45 vs 306 samples; the fine grid keeps neighbours 2x closer, so the two CVs are not the same difficulty |
| cv_kappa0_rel_median | frog | 0.045256 | 0.0387216 | false | same LOO difference as above |
| cv_qv_rmse_median | frog | 0.0112785 | 0.00751778 | false | same LOO difference as above |
| cv_density_rel_median | frog | 0.173322 | 0.126298 | false | spot-density regression error, same LOO difference |
| e2e_complete_steady_rate | frog | 24/24 | 23/24 | true | full residual <= 1e-11 for all 24 held-out runs; same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_legacy_success_rate | frog | 24/24 | 23/24 | true | published legacy rule (rotation-labelled modes excluded from max Re lambda); same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_strict_task_point_rate | frog | 5/6 | 4/6 | true | strict task success on the 6 runs at the two task parameter points; same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_reference_morphology_match | frog | 13/24 | 13/24 | true | observed type equals the PDE reference majority type (surrogate target at random points); same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_spectral_pass_rate | frog | 24/24 | 23/24 | true | no positive sampled eigenvalue (rotation included); legacy values are unvalidated; same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_relative_equilibrium_rate | frog | 0/24 | 0/24 | true | rotation-removed residual <= 1e-11 but full residual above; same 8 points x 3 seeds, but different run dates/loads for wall time |
| steps_median_legacy_success | frog | 15062.5 | 13625 | true | integration steps are machine-independent; compare only with the same time cap |
| heldout_type_accuracy | frog | 8/8 | 8/8 | true | 8 held-out points, predicted type vs target/PDE-majority reference; old/new predictions agree on 8/8 points; mixed PDE reference points: 0 |
| cv_type_accuracy | sphere5 | 0.888889 | 0.928105 | false | leave-one-grid-point-out with 45 vs 306 samples; the fine grid keeps neighbours 2x closer, so the two CVs are not the same difficulty |
| cv_kappa0_rel_median | sphere5 | 0.0541394 | 0.0243137 | false | same LOO difference as above |
| cv_qv_rmse_median | sphere5 | 0.0105628 | 0.0069845 | false | same LOO difference as above |
| cv_density_rel_median | sphere5 | 0.226673 | 0.105101 | false | spot-density regression error, same LOO difference |
| e2e_complete_steady_rate | sphere5 | 0/24 | 0/24 | true | full residual <= 1e-11 for all 24 held-out runs; same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_legacy_success_rate | sphere5 | 12/24 | 10/24 | true | published legacy rule (rotation-labelled modes excluded from max Re lambda); same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_strict_task_point_rate | sphere5 | 0/6 | 0/6 | true | strict task success on the 6 runs at the two task parameter points; same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_reference_morphology_match | sphere5 | 22/24 | 23/24 | true | observed type equals the PDE reference majority type (surrogate target at random points); same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_spectral_pass_rate | sphere5 | 10/24 | 6/24 | true | no positive sampled eigenvalue (rotation included); legacy values are unvalidated; same 8 points x 3 seeds, but different run dates/loads for wall time |
| e2e_relative_equilibrium_rate | sphere5 | 13/24 | 10/24 | true | rotation-removed residual <= 1e-11 but full residual above; same 8 points x 3 seeds, but different run dates/loads for wall time |
| steps_median_legacy_success | sphere5 | 27312.5 | 20937.5 | true | integration steps are machine-independent; compare only with the same time cap |
| heldout_type_accuracy | sphere5 | 6/8 | 5/8 | true | 8 held-out points, predicted type vs target/PDE-majority reference; old/new predictions agree on 7/8 points; mixed PDE reference points: 0 |

### 5.2 计算成本（按同一成功口径拆分）

| 组 | 旧口径成功 | 新口径成功 | 墙钟中位：全部/旧口径成功/旧口径失败 (s) | 步数中位：全部/旧口径成功/旧口径失败 | host · load |
|---|---|---|---|---|---|
| ai_generation:frog:ai | 24/24 | 13/24 | 678.284 / 678.284 /  | 15062.5 / 15062.5 /  | JCnoLAPTOP · interleaved_with_pde |
| ai_generation:frog:pde | 22/24 | 22/24 | 1166.07 / 1166.07 / 1963.48 | 25875 / 25875 / 44375 | JCnoLAPTOP · interleaved_with_pde |
| ai_generation:sphere5:ai | 12/24 | 0/24 | 279.482 / 191.399 / 310.406 | 45562.5 / 27312.5 / 51500 | TorinoUta19.local · concurrent_L6_generation |
| ai_generation:sphere5:pde | 6/24 | 0/24 | 372.986 / 274.929 / 374.206 | 65000 / 44500 / 65000 | TorinoUta19.local · concurrent_L6_generation |
| ai_generation_fine:frog:ai | 23/24 | 12/24 | 520.818 / 517.2 / 1249.31 | 13812.5 / 13625 / 51500 | JCnoLAPTOP · ai_only |
| ai_generation_fine:frog:pde | 22/24 | 22/24 | 1166.07 / 1166.07 / 1963.48 | 25875 / 25875 / 44375 | JCnoLAPTOP · reused_old_baseline |
| ai_generation_fine:sphere5:ai | 10/24 | 0/24 | 160.281 / 74.5348 / 174.103 | 51500 / 20937.5 / 51500 | TorinoUta19.local · ai_only |
| ai_generation_fine:sphere5:pde | 6/24 | 0/24 | 372.986 / 274.929 / 374.206 | 65000 / 44500 / 65000 | TorinoUta19.local · reused_old_baseline |

注：本表墙钟统一用 `total_s`（含 setup）；旧文中的 673 s / 1159 s 用的是 `solve_s`，是同一批运行，差异只来自字段选择。
注：`evaluation:*` 组的新口径成功要求形态与任务/参考形态一致；随机参数点上没有“任务要求”，用 PDE 参考形态作替代目标，因此它衡量的是泛化一致性，不是任务要求。任务点上的严格成功率见 5.1 的 `e2e_strict_task_point_rate`。

### 5.3 AI / PDE 成对比较（同一参数点、同一种子）

| 对比 | 成对样本 | 步数比中位（旧口径） | 墙钟比中位（旧口径） | 成对样本（新口径） | 步数比中位（新口径） | 步数比可用 | 墙钟比可用 |
|---|---|---|---|---|---|---|---|
| ai_generation:frog:ai_vs_pde | 24 | 0.485283 | 0.472505 | 13 | 0.331731 | true | true |
| ai_generation:sphere5:ai_vs_pde | 12 | 0.500485 | 0.550297 | 0 |  | true | true |
| ai_generation_fine:frog:ai_vs_pde | 23 | 0.509346 | 0.43583 | 12 | 0.501548 | true | false |
| ai_generation_fine:sphere5:ai_vs_pde | 10 | 0.464712 | 0.260586 | 0 |  | true | false |

**公平性问题清单**

1. 新模型的 PDE 行是从旧 `evaluation.json` 复用的 48 条记录（逐字段相同，见 manifest 检查）；新模型 AI 行是单独跑的（青蛙 20 进程、球面 8 进程），因此 `summary.md` 里的 2.2x / 2.3x 墙钟加速比不成立。
2. 旧模型青蛙那组 AI 与 PDE 在同一台笔记本交错运行、负载相同，673 s vs 1159 s 可用；对应的成对步数比见上表。
3. 旧/新模型之间墙钟不可比（旧模型与 PDE 交错、新模型单独跑）；模型对比只能看积分步数。
4. L5 评测从未做 orient：24 次运行没有一个完整稳态，全部是相对平衡态；因此 L5 的成功率不能与球面 L6 形式任务（完整稳态）直接比较。
5. 留一法类型准确率 64%→86%（青蛙）、89%→93%（球面）不可直接比较：旧网格 45 点、新网格 306 点，去掉一点后邻点间距差 2 倍，难度不同。
6. 同一批 8 个留出点上新旧预测只有 1 个不同（球面 random5：旧判条纹、新判孔洞），所以“新模型更强”主要不是分类器变化，而是训练数据与形态先验的变化。
7. 任务点上的严格成功率（完整稳态 + 无正模态 + 形态 + 预算）只有 6 次运行，不能与 24 次留出评测混用同一个分母。

## 6. 训练标签与数据质量

| 数据 | 行数 | 收敛到 1e-11 | 无正模态 | 有正模态 | 谱未检查 | 空标签（均匀态） | 预算 |
|---|---|---|---|---|---|---|---|
| ai_generation · frog | 45 | 32 | 31 | 1 | 13 | 11 | 1e5 |
| ai_generation · sphere5 | 45 | 15 | 14 | 0 | 31 | 14 | 1e5 |
| ai_generation_fine · frog | 306 | 257 | 253 | 4 | 49 | 73 | 1e5 |
| ai_generation_fine · sphere5 | 306 | 90 | 83 | 0 | 223 | 83 | 1e5 |

- 训练集用 t=1e5 预算、评测/任务用 t=2e5；球面 L5 的 306 条标签只有 90 条达到 1e-11（多为斑点，条纹/迷宫停在转动漂移或重排），这些未收敛标签仍被用于训练。
- 青蛙新数据中有 4 个标签态存在正采样模态；用不稳定态做形态回归目标需要在报告中说明。
- 空标签全部是 uniform 态（`labels={}`），不是缺失标签。

## 7. I03 技术报告与提交包应采用的数字

### 7.1 任务结果（四组正式候选）

| 任务 | 采用结果 | 完整残差 | 谱状态 | 说明 |
|---|---|---|---|---|
| 任务1 球面斑点 | 45 斑，MAT `44f5931a…` | 8.423e-15 | numerical_check_passed（I02 三配置，max Re λ=-6.231e-6） | 完整稳态；谱检查含旋转类模态 |
| 任务1 球面条纹 | 七条带 seed2，MAT `ef132d82…` | 2.437e-14 | numerical_check_passed（I02 三配置，max Re λ=-1.478e-6） | **不要**再用六条带 seed0（可靠正倾斜模态 +1.6654e-6） |
| 任务2 青蛙斑点 | 22 斑，MAT `19a92c65…` | 2.171e-14 | not_checked（无绑定谱证书） | 旧单 shift 记录 −1.03e-4 属于 I01 之前的场 |
| 任务2 青蛙条纹 | 5 段迷宫，MAT `aa99f137…` | 1.006e-14 | not_checked（无绑定谱证书） | 旧单 shift 记录 −1.55e-4 属于 I01 之前的场 |

### 7.2 稳健性（33 / 23 个种子，题给参数）

- 青蛙斑点：31/33 = 完整稳态且未发现正模态。失败：seed22 鞍点（+3.2e-5）、seed25 预算耗尽。
- 青蛙条纹：收敛 33/33、谱全负 33/33，但**目标条纹形态只有 31/33**（seed1、seed21 是斑点态）；不能写“33/33 目标条纹生成成功”。
- 球面斑点：完整稳态 17/23；3 个有正旋转模态（seed13/16/21）；无正模态候选 14/23；6 个预算失败。14/23 是有限采样候选率，不是数学稳定率。
- 球面条纹：完整稳态 22/23；13 个六条带有正倾斜模态 +1.67e-6，9 个七条带无正模态；只有 9/23 同时满足“完整稳态 + 无正模态”。其中 seed2 有 I02 证书；其余 8 个七条带种子（3/8/11/12/18/19/20/22）只有旧单 shift 证据，不能用 seed2 的证书代替独立验收。

### 7.3 AI 对比

- 青蛙（8 点 × 3 种子）：旧模型 AI 24/24（全部完整稳态、谱全负）、新模型 23/24、普通流程 22/24。
- 青蛙成本：步数中位 旧 AI 15062、新 AI 13812、普通 25875（AI 少 42–47%）；旧模型那组同负载墙钟约 1.7x（solve_s 673/1159，total_s 678/1166）可用；新模型只能写步数。
- 球面 L5：旧 AI 12/24、新 AI 10/24、普通 6/24 都是“相对平衡态 + 旧谱口径”；按本阶段严格口径（相对平衡 + 无正模态）为 10/24、6/24、4/24，且 0 个完整稳态。
- 留一法：青蛙 64%→86%、球面 89%→93%，必须注明两者网格密度与留一难度不同。
- 端到端泛化：8 个留出点上新旧预测只有球面 random5 不同；青蛙两模型在孔洞区都收敛到迷宫而不是孔洞，这是形态先验问题，不是求解器失败。

## 8. 证据缺口与后续建议

1. 稳健性扫描没有保存 MAT/场 → `file_validity=not_saved`；3 个正旋转模态与 13 个六条带正模态无法在本阶段复核。建议对 seed13/16/21 与六条带代表态补 I02 式审计。
2. 旧谱值都是单 shift、k≈8、无特征对残差：`unstable_positive_mode` 是保守判定；升级为“可信正模态”需要 I02 的三配置 + 特征对残差 + 小扰动验证。
3. 青蛙两个正式候选缺少绑定到 I01 MAT 的谱检查（Newton 修正 ≤3e-8，但不能当作证明）；建议给 `scripts/i02_stability.py` 增加 frog 分支后补齐。
4. L5 评测没有 orient，因此没有完整稳态；若要在统一口径下报告 L5，需要重跑 orient 版本。
5. 训练标签与评测的 t_max 不同（1e5 vs 2e5），训练目标含大量未收敛/相对平衡态；建议只用 ≤1e-11 的稳态做形态回归，或在标签中记录收敛等级。
6. 提交包/README 中的“24/24 收敛且稳定”“33/33 条纹”“22/23 条纹”必须按本报告改写；本轮不修改报告与提交包。

## 9. 可复现性与检查

- 记录数：1011；检查项 25 条，失败 0 条；源问题 0 条。
- [x] denominators:robustness:frog_spots 
- [x] denominators:robustness:frog_stripes 
- [x] denominators:robustness:sphere6_spots 
- [x] denominators:robustness:sphere6_stripes 
- [x] denominators:formal_candidates 
- [x] denominators:evaluation:ai_generation:frog:ai 
- [x] denominators:evaluation:ai_generation:frog:pde 
- [x] denominators:evaluation:ai_generation:sphere5:ai 
- [x] denominators:evaluation:ai_generation:sphere5:pde 
- [x] denominators:evaluation:ai_generation_fine:frog:ai 
- [x] denominators:evaluation:ai_generation_fine:frog:pde 
- [x] denominators:evaluation:ai_generation_fine:sphere5:ai 
- [x] denominators:evaluation:ai_generation_fine:sphere5:pde 
- [x] denominators:training:ai_generation:frog 
- [x] denominators:training:ai_generation:sphere5 
- [x] denominators:training:ai_generation_fine:frog 
- [x] denominators:training:ai_generation_fine:sphere5 
- [x] legacy_count_reproduced:robustness:frog_spots published=31 recomputed=31
- [x] legacy_count_reproduced:robustness:frog_stripes published=33 recomputed=33
- [x] legacy_count_reproduced:robustness:sphere6_spots published=17 recomputed=17
- [x] legacy_count_reproduced:robustness:sphere6_stripes published=22 recomputed=22
- [x] baseline_rows_reused_identically 48/48 pde rows reused
- [x] stability_certificates_bound 3 certificates
- [x] i01_mats_accepted all valid candidate MATs accepted
- [x] no_parse_errors 0 source problems

复现命令：`python scripts/i04_evaluate.py --out results/i04_evaluation`（目录必须不存在）。所有分母、失败样本和源文件哈希见 `success_breakdown.csv`、`evaluation_records.json` 与 `run_manifest.json`。
