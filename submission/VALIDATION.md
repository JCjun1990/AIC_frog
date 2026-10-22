# 本次交付工程验收（2026-10-22）

基准 main：37a4d25f0aa30cb30be47059715130add0aac5d4。队名未提供，仍为交付草稿，不自动上传。

- 改动前 83 项单元测试通过；修复后 89 项通过。
- 新增回归覆盖真实四组证书绑定、正模态/残差/收敛标记篡改、Windows 原始 ZIP 文件名、目录逃逸、符号链接、重名和拒绝覆盖。
- 仓库外全新目录独立解包：CRC、文件清单与 SHA256、包内代码导入、4 MAT 完整回读、反演一致性、路径卫生及新报告页数/关键数字全部通过。
- 四组正式 MAT、两个模型、任务3数值结果与原草稿逐字节相同；正式 summary/证书保留原 ZIP 字节（包括原始换行）。
- 新仓库 .gitattributes 固定一般文本 LF；冻结任务载荷禁止 Git 文本转换。没有修改系统 Git 设置。
- 完整 robust PDE 残差：sphere spots 8.423e-15，sphere seven-band stripes 2.437e-14，frog spots 2.171e-14，frog stripes 1.006e-14，均不做转动投影。
- 四组为三配置有限数值谱证据；青蛙各40特征对。未移位 LR 失败保留，既有特征对元数据逐对检查。此轮未重新求完整谱，包内没有原始特征向量，不能称全谱证明。
- 反演 MAT 重新计算残差约9.061e-14；估计值及原始记录未改。目标函数带不是校准置信区间。
- 新两页报告为 report/初赛技术报告_交付草稿_v2.pdf；已渲染检查。旧 report/初赛技术报告.pdf 保留，仅作历史证据。
- 8 个无引用的系统/网络/远程维护脚本移除，未执行；真正的 overnight_fine.bat 数值训练入口保留，但本轮未运行。
- git diff --check 通过。历史 AI 摘要原始数字保留，自动重建脚本及 CSV 明确区分旧标记与未投影残差。

## 实测环境和兼容边界

Python 3.12.5；numpy 2.3.2；scipy 1.17.0；torch 2.14.1+cpu；
robust-laplacian 1.1.0；scikit-learn 1.7.1；PyMuPDF 1.27.1。
复用既有 Python 环境，本轮未安装依赖。
两个附带 MLP 的历史 sklearn 版本与本机版本不同；加载警告保留。只核验本机可加载及预测，不保证所有 sklearn 版本的 pickle 兼容。
PDF 检查/新草稿生成额外依赖 PyMuPDF，见 code/requirements-delivery.txt（不属于 PDE 核心依赖）。
报告生成还需复现者显式提供可用 CJK 字体；新 PDF 内已嵌入子集字体。

3份历史 JSON 使用 Python 的 NaN/Infinity 扩展：任务3 summary 及两个模型 dataset。
为保持已有结果/标签原件，未替换其数值；这3份不是严格 RFC JSON。
evidence/json_schema_audit.json 明确列出文件、SHA 与非有限标记；严格 JSON 消费者须显式适配。
新生成的清单/证据 JSON 都是严格 JSON。CSV 列数、表头及记录数已检查。

## 未复跑和上传阻塞

未重训 MLP，未修改 PDE、正式 MAT 或反演参数；未复跑昂贵的从头训练/全量生成/全谱搜索。
独立解包复算在 Windows CPU 实际执行；Linux 命令和路径仅经静态检查。
正式队名、命名后的报告与最终上传验收仍待完成。当前不得声称“可直接上传”。
原 ZIP 保留于基准 Git 提交和完整本地安全备份；新版 ZIP 使用当前证据重新构建。
最终 ZIP SHA256 与独立验收结果见仓库根 DELIVERY_VALIDATION.json（外置，避免 ZIP 自引用哈希）。

当前仓库草稿 ZIP：`submission/drafts/待填_初赛_AI-MECH-2026-002_result.zip`。
`submission/待填_初赛_AI-MECH-2026-002_result.zip` 是保留原字节的历史旧稿，内部证据较旧，不作为本次验收包。
