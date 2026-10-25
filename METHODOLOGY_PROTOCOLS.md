# F8/F9：版本化质量协议

历史数据、v1/v2 模型和正式反演参数保持原样。本次新增的质量筛选不追溯改变旧训练协议。

## 标签质量

`turing_rd/quality.py` 定义 `endpoint-label-quality-v1`。`finite_time_state` 表示有效有限时长终态，`complete_steady` 要求记录的完整残差达标；`spectrum_checked_pass` 还要求固定政策规定的求谱配置、数量、逐特征对残差和符号均通过。`spectrum_positive` 表示发现可靠正模态，`spectrum_insufficient` 表示证据尚不足。完整稳态是终态的子集，这些标志不是互斥类别。

历史 AI 数据仅有原终态的残差标量，不能据此认证 float32 缓存回读场。历史谱摘要缺少要求的特征对数量或逐对证据时，不升级为新协议通过。详见独立 `results/methodology_stage6_20261025T121811/` 统计。球面测试的 16 个完整稳态均为均匀态，不证明非平凡斑图泛化。

未来使用 `scripts/train_quality.py --data <数据.json> --out <新目录> --quality-policy complete_steady` 可仅审查筛选结果。默认不训练；明确添加 `--execute-training` 才训练新名称权重。可选策略为 `all_endpoints`、`complete_steady`、`steady_and_spectrum`。禁止用旧模型目录作为输出。

## 反演参考库

`is_steady()` 保留 `legacy-rate-v1` 原语义，仅用于历史回读。新 `reference_usable()` 和 `scripts/invert_target.py` 默认 `full-pde-v1`：在实际返回的 float64 双变量场上，以原 robust 算子测量

`R = sqrt((||Ru||²_M + ||Rv||²_M)/(2*area)) / F <= 1e-11`。

状态哈希、双变量 RMS、公式版本、参数、几何身份及 dtype 一起记录。低 `dv/dt`、PTC/Newton 成功标志和谱摘要都不能替代该残差。严格流程拒绝旧缓存、参考点不足时的宽松回退，以及最终代表解的未验证谱筛选。显式选择 `--reference-protocol legacy-rate-v1` 才能重放旧筛选语义。所有新计算必须另设 `--out`。

历史 294 条损失仍可复算原损失曲面；130/176/46 仅为汇总交叉核对。缺少逐条场和 rate，不能重建严格稳态分类或严格参数偏移。“严格证据覆盖为零”不等于“实际稳态为零”。正式 F=0.0277124、k=0.0600398 不变。

## 最佳匹配谱证据

`scripts/best_match_spectrum.py --out <新目录> --seconds 180` 只对现有 MAT 做三组有限谱求解，保存 12+20+8 个特征对及向量，并绑定 MAT、网格和算子 SHA256。硬时间预算最高 300 秒，超时或失败不产生通过结论。`--verify <证书.json>` 重新回读 MAT、重算完整残差及保存向量的 `Jw-lambda*w`，不重新求谱。

本轮新增证据位于 `results/methodology_stage6_20261025T121811/best_match_spectrum/`，交付副本在 `submission/evidence/methodology/best_match_spectrum/`。原最佳匹配摘要保留，新增证据不改写历史。验收明确区分 PDE 回读、保存谱元数据验证、保存向量残差重算和实际新求谱。三个 shift-invert 配置不构成全谱数学证明。

最佳匹配为固定 F/k 的 `cvt_3` 多初值筛选，初始 20 斑、最终 16 斑，特征距离 J 从约 1.950 降至 1.554。未开展非 AI 等预算消融，AI 独立因果贡献仍未证明。
