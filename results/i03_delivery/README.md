# results/i03_delivery — I03 交付目录指南

本目录是 I03（初赛技术报告、正式结果与提交包）的全部产物。**未修改 I01/I02/I04 的任何结果。**

## 交付状态

**工程交付验收通过（草稿）。** 唯一未完成项：队名仍为 `待填`，补齐后需重新生成报告与 ZIP。
完成报告见 `I03_DELIVERY_REPORT.md`。

## 目录

| 路径 | 内容 | 状态 |
| --- | --- | --- |
| `candidates.json` | 四组正式候选：实验 id、MAT 路径/SHA256、完整残差、形态、谱状态、证书路径与哈希、选择策略 | 最终 |
| `frog_stability/` | 青蛙谱检查 **v1**：直接 LR 搜索失败记录（斑点 2/8、条纹 0/8），保留作证据 | 已保留 |
| `frog_stability_v2/` | 青蛙谱检查 **最终版**：三 shift-invert 配置 + `frog_stability_certificate.json`（两组） | 最终 |
| `figures/` | 四份 MAT 的终态渲染（`task*.png`）与报告四联图 `report_panel.png` | 最终 |
| `report/` | `初赛技术报告_I03_草稿.docx`（源）、`.pdf`（2 页成品）、`report_data.json`（数字来源） | 草稿 |
| `package/` | 提交包 **v1**：首次独立验收 FAIL（README 未进清单、JSON 含本机路径） | 已保留作迭代证据 |
| `acceptance/` | v1 的验收记录（FAIL） | 已保留 |
| `package_v2/` | 提交包 **最终版**：`待填_初赛_AI-MECH-2026-002_result/` + `.zip`（72 文件，11.1 MB） | 最终（占位队名） |
| `acceptance_v2/` | v2 的独立解包验收（**PASS 21/21**） | 最终 |
| `I03_DELIVERY_REPORT.md` | A–I 交付报告 | 最终 |

## 复现命令（仓库根目录，`.venv` Python）

```bash
# 1) 青蛙两组独立谱检查（约 1 分钟）
python scripts/i03_frog_stability.py --out results/i03_delivery/frog_stability_v2

# 2) 按哈希定位四组正式候选并绑定证书
python scripts/i03_candidates.py --out results/i03_delivery/candidates.json

# 3) 从提交 MAT 渲染图
python scripts/i03_render_figures.py

# 4) 两页报告 DOCX（用 bundled python-docx）→ PDF（LibreOffice Kit）
<bundled-python> scripts/i03_report_docx.py
<node> <libreoffice-kit>/cli.js convert --input results/i03_delivery/report/初赛技术报告_I03_草稿.docx \
       --output results/i03_delivery/report/初赛技术报告_I03_草稿.pdf

# 5) 打包（新目录，不覆盖）
python scripts/i03_package.py --team 待填 --out results/i03_delivery/package_v2

# 6) 仓库外独立解包验收
python scripts/i03_verify_package.py \
  --zip results/i03_delivery/package_v2/待填_初赛_AI-MECH-2026-002_result.zip \
  --out results/i03_delivery/acceptance_v2 \
  --report results/i03_delivery/report/初赛技术报告_I03_草稿.pdf
```

## 上传前必须完成

1. 提供真实队名；
2. 修改 `scripts/i03_report_docx.py` 顶部 `TEAM` 并重建 DOCX/PDF；
3. 用 `--team <队名> --out .../package_final` 重新打包并重跑验收；
4. 在报名系统【赛题结果文件】上传 ZIP、【文档类文件】上传 PDF（报告不放入 ZIP）。
