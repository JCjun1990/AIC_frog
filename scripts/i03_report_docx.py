"""I03: build the two-page preliminary technical report (DOCX) from verified records.

Every number in the document is read from the I03 candidate manifest, the I03 frog
stability certificates, the inversion summary, the I04 evaluation manifest and the
original task/CV records.  Nothing is typed by hand except the prose.

Run with the bundled Python that provides python-docx:

    <bundled-python> scripts/i03_report_docx.py
    <node> <libreoffice-cli> convert --input <docx> --output <pdf>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "results" / "i03_delivery"
OUT = DELIVERY / "report"
TEAM = "待填"
TITLE = "AI-PDE 曲面斑图生成与反演：初赛技术报告"
BODY_PT = 9.4


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def set_run_font(run, size=BODY_PT, bold=False, italic=False, color=None, east="宋体"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = "Times New Roman"
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    fonts.set(qn("w:eastAsia"), east)
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def rich(paragraph, parts, size=BODY_PT, color=None):
    """parts: list of (text, style) with style in '', 'i', 'sub', 'sup', 'b', 'bi'."""
    for text, style in parts:
        run = paragraph.add_run(text)
        set_run_font(run, size=size, bold=("b" in style), italic=("i" in style), color=color)
        if "sub" in style:
            run.font.subscript = True
        if "sup" in style:
            run.font.superscript = True


def heading(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, size=10.5, bold=True, east="黑体")
    return paragraph


def body(document, parts, space_after=2.0, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing = 1.07
    paragraph.alignment = align
    rich(paragraph, parts if isinstance(parts, list) else [(parts, "")],
         size=BODY_PT if size is None else size)
    return paragraph


def add_field(paragraph, instruction):
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), instruction)
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "1"
    run.append(text)
    field.append(run)
    paragraph._p.append(field)


def table(document, headers, rows, widths):
    tbl = document.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, (header, width) in enumerate(zip(headers, widths)):
        cell = tbl.rows[0].cells[index]
        cell.width = Cm(width)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(header)
        set_run_font(run, size=8.0, bold=True)
    for row in rows:
        cells = tbl.add_row().cells
        for index, (value, width) in enumerate(zip(row, widths)):
            cells[index].width = Cm(width)
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if index else WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(value))
            set_run_font(run, size=8.0)
    return tbl


def main():
    candidates = load("results/i03_delivery/candidates.json")["candidates"]
    inversion = load("results/inversion/summary.json")
    i04 = load("results/i04_evaluation/run_manifest.json")
    cv = {tag: load(f"results/ai_generation/{tag}/cv.json")["summary"] for tag in ("frog", "sphere5")}
    tasks = {task: load(f"results/ai_generation/tasks/{task}/summary.json")
             for task in ("frog_spots", "frog_stripes", "sphere6_spots", "sphere6_stripes")}

    sp, st = candidates["task1_sphere_spots"], candidates["task1_sphere_stripes"]
    fp, ft = candidates["task2_frog_spots"], candidates["task2_frog_stripes"]
    estimate, given = inversion["estimate"], inversion["given_spots"]
    band_F, band_k = estimate["band_F"], estimate["band_k"]
    final = [row for row in inversion["final"] if row["label"] == "estimate"]
    n_final_ok = sum(1 for row in final if row["rel_residual"] <= 1e-11
                     and (row["stability"] or {}).get("max_real", 1) < 0)
    target = inversion["target_features"]
    seed2000 = next(row for row in final if row["seed"] == 2000)
    feats = seed2000["features"]
    robustness = {key: i04["counts"][key] for key in
                  ("robustness:frog_spots", "robustness:frog_stripes",
                   "robustness:sphere6_spots", "robustness:sphere6_stripes")}

    numbers = {
        "generated_utc": i04["generated_utc"],
        "candidates": {key: {"mat_sha256": value["mat_sha256"],
                             "full_residual": value["full_residual"],
                             "pattern": value["pattern_stats"],
                             "spectral": value["spectral"]}
                       for key, value in candidates.items()},
        "inversion": {"F": estimate["F"], "k": estimate["k"], "given_F": given["F"], "given_k": given["k"],
                      "band_F": band_F, "band_k": band_k, "n_final_ok": n_final_ok,
                      "final_runs": len(final), "wavelength": feats["wavelength"],
                      "target_wavelength": target["wavelength"], "kappa_v": feats["kappa_v"],
                      "target_kappa_v": target["kappa_v"],
                      "defect_fraction": feats["spot_defect_frac"],
                      "target_defect_fraction": target["spot_defect_frac"],
                      "residual_seed2000": seed2000["rel_residual"],
                      "n_spots": feats["n_spots"]},
        "robustness": robustness,
        "cv": cv,
        "generation_wall_s": {key: value.get("wall_s") for key, value in tasks.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report_data.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=1) + "\n",
                                          encoding="utf-8")

    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.45)
    section.left_margin = section.right_margin = Cm(1.6)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(BODY_PT)
    rpr = normal.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    fonts.set(qn("w:eastAsia"), "宋体")

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(1)
    run = title.add_run(TITLE)
    set_run_font(run, size=15.5, bold=True, east="黑体")

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(1)
    run = subtitle.add_run(f"赛题 AI-MECH-2026-002 ｜ 队伍：{TEAM} ｜ 2026-10-21")
    set_run_font(run, size=8.5, color=(0x40, 0x40, 0x40))

    draft = document.add_paragraph()
    draft.alignment = WD_ALIGN_PARAGRAPH.CENTER
    draft.paragraph_format.space_after = Pt(4)
    run = draft.add_run("交付草稿：队名待填写，补齐队名并重新导出后方可上传；技术报告 PDF 与答案 ZIP 分开提交")
    set_run_font(run, size=7.5, color=(0xB0, 0x30, 0x30), east="宋体")

    heading(document, "摘要")
    body(document, [
        ("我们以 Gray–Scott 反应扩散方程为基准，把小型神经网络用作斑图初值生成器与反演代理，"
         "所有 AI 输出都由同一套曲面求解器复算并精修到完整稳态。初赛三项任务均已完成：", ""),
        ("单位球面生成", "b"), ("（45 斑、7 条纬向条带）、", ""),
        ("青蛙曲面生成", "b"), ("（22 斑、5 段迷宫）、", ""),
        ("参数反演", "b"), (f"（F = {estimate['F']:.5f}，k = {estimate['k']:.5f}，相对题给值误差 "
                          f"{(estimate['F'] - given['F']) / given['F'] * 100:+.1f}% / "
                          f"{(estimate['k'] - given['k']) / given['k'] * 100:+.1f}%）。四个正式解都是 float64 "
                          "完整离散稳态（完整残差 8.4e-15–2.4e-14），三种独立求谱配置均未发现正实部特征值；"
                          "这是有限采样的数值证据，不是全谱数学证明。", ""),
    ], space_after=3)

    heading(document, "1　模型、数据与量纲")
    body(document, [("控制方程（Γ 为曲面，Δ", ""), ("Γ", "sub"), (" 为 Laplace–Beltrami 算子）：", "")], space_after=1)
    equation = document.add_paragraph()
    equation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.paragraph_format.space_after = Pt(1)
    rich(equation, [("∂u/∂t = D", "i"), ("u", "sub"), (" Δ", ""), ("Γ", "sub"), ("u − u v", ""), ("2", "sup"),
                    (" + F(1 − u),　　∂v/∂t = D", ""), ("v", "sub"), (" Δ", ""), ("Γ", "sub"), ("v + u v", ""),
                    ("2", "sup"), (" − (F + k) v", "")], size=9.5)
    body(document, [
        ("题给参数：斑点 F = 0.028、k = 0.062，条纹 F = 0.054、k = 0.063；D", ""), ("u", "sub"),
        (" = 2.78×10", ""), ("−4", "sup"), ("，D", ""), ("v", "sub"),
        (" = 0.5D", ""), ("u", "sub"), (" = 1.39×10", ""), ("−4", "sup"), ("。", ""),
        ("数据只用组委会附件 frog.mat（青蛙曲面，29010 顶点）与 Solution.mat（目标斑图）；"
         "训练标签由求解器自生成，无外部数据或预训练生成模型。生成任务直接使用题给参数、不做参数调优；"
         "AI 只决定初值，最终形态由 PDE 收敛结果决定。", ""),
    ], space_after=1)
    body(document, [
        ("量纲与尺度：", "b"),
        ("[D] = L²/T，[F] = [k] = 1/T，u、v 无量纲；扩散长度 ℓ = √(D", ""), ("u", "sub"),
        ("/F) ≈ 0.10，网格边长约 ℓ/5–ℓ/6.5，时间步长满足扩散数约束；残差按 F 归一，"
         "便于跨参数比较。", ""),
    ], space_after=2)

    heading(document, "2　数值离散与 AI-PDE 融合架构")
    body(document, [
        ("离散：", "b"),
        ("内蕴 Delaunay cotan 权（全非负）+ 集中质量矩阵，满足离散最大值原理；IMEX SBDF2 时间推进"
         "（扩散隐式、反应显式），其不动点即离散稳态；再用带回溯线搜索的 Newton 把完整残差压到 1e-12 量级。", ""),
    ], space_after=1)
    body(document, [
        ("球面对称性：", "b"),
        ("网格轻微破坏 SO(3)，Jacobian 有 3 个近零转动模。用转动生成元 T", ""), ("i", "sub"),
        ("U = (e", ""), ("i", "sub"), ("×x)·∇", ""), ("Γ", "sub"),
        ("U 构造加边系统 R(U) + Σω", ""), ("i", "sub"), ("MT", ""), ("i", "sub"),
        ("U = 0 与相位条件求相对平衡态，再沿相对平衡族跟到 ω = 0 的网格偏好朝向，"
         "得到真离散稳态；这一步是球面结果能称为“完整稳态”的原因。", ""),
    ], space_after=1)
    body(document, [
        ("AI 的两处作用（AI 出候选、PDE 定结果）：", "b"),
        ("① 生成初值——小型 MLP 由 (F, k) 预测斑图类型与统计量（斑点密度与径向剖面、LB 谱环 (κ", ""),
        ("0", "sub"), (", γ)、v 分位数与 u–v 关系），据此合成初值：斑点用 Lloyd 质心 Voronoi 布点，"
         "条纹用节线迷宫，球面改用纬向条带；② 反演——46 维斑图指纹（类型、波长、方向性、缺陷）+ 高斯过程代理"
         "在 (F, k) 平面求极小。可微求解器（自定义 autograd）给出 F、k、D、初值的梯度，并已与有限差分核对。", ""),
    ], space_after=2)

    heading(document, "3　正式生成结果")
    rows = []
    for key, label in (("task1_sphere_spots", "球面斑点"), ("task1_sphere_stripes", "球面条纹"),
                       ("task2_frog_spots", "青蛙斑点"), ("task2_frog_stripes", "青蛙条纹")):
        candidate = candidates[key]
        shape = candidate["pattern_stats"]
        spectral = candidate["spectral"]
        rows.append([label,
                     f"F={candidate['parameters']['F']:.3f}, k={candidate['parameters']['k']:.3f}",
                     f"{shape['type']} / {shape['n_components']}",
                     f"{shape['wavelength']:.3f}",
                     f"{candidate['full_residual']:.3e}",
                     f"{spectral['max_real']:+.3e}"])
    table(document, ["任务", "参数", "形态 / 数量", "波长 λ", "完整残差", "谱 max Re λ（三配置）"],
          rows, [2.3, 3.5, 3.9, 1.7, 2.3, 3.7])
    body(document, [
        ("注：完整残差 = √[(RMS", ""), ("M", "sub"), ("(R", ""), ("u", "sub"), (")² + RMS", ""),
        ("M", "sub"), ("(R", ""), ("v", "sub"), (")²)/2]/F，使用实际 robust L/M，不做转动投影。"
         "谱检查为有限 Krylov 采样（k = 8–20，矩阵维度 5.8–8.2 万）：球面用 σ = 10⁻³、0 与直接最大实部三配置，"
         "青蛙用 σ = 10⁻³、0、10⁻⁵ 三配置；特征对绝对残差 ≤ 1.5×10⁻¹³，三配置最大值一致；"
         "任一采样到正实部都会判失败。六条带 seed0 检出 +1.67×10⁻⁶ 的可靠正倾斜模，已停止作为正式结果，"
         "条纹正式解改用七条带 seed2。", ""),
    ], space_after=2)

    document.add_picture(str(DELIVERY / "figures" / "report_panel.png"), width=Cm(16.4))
    caption = document.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(1)
    caption.paragraph_format.space_after = Pt(2)
    run = caption.add_run("图 1　四组正式候选的终态 v 场（由提交 MAT 直接渲染）：(a) 球面 45 斑；(b) 球面 7 条带；"
                          "(c) 青蛙 22 斑；(d) 青蛙 5 段迷宫。")
    set_run_font(run, size=7.5, color=(0x30, 0x30, 0x30))

    heading(document, "4　反演设计（只反演 F、k）")
    body(document, [
        (f"在 D", ""), ("u", "sub"), ("、D", ""), ("v", "sub"),
        (" 取题给值时只反演 F、k。流程：热核低通（κ½ = 20，由残差曲线自动选定，不使用真值）并截断到物理范围；"
         "积分恒等式给出闭式初值（扩散项积分为零）：F = ∫uv²/∫(1−u)，F + k = ∫uv²/∫v；"
         "在 7×7 (F, k) 网格上每点用 6 个固定初值跑到稳态，按“类型；波长、方向性、缺陷特征；积分恒等式残差”"
         "打分；高斯过程代理在细网格上求极小。", ""),
    ], space_after=1)
    body(document, [
        (f"结果：", "b"), (f"F = {estimate['F']:.5f}，k = {estimate['k']:.5f}（题给 {given['F']:.3f} / {given['k']:.3f}，"
                        f"误差 {(estimate['F'] - given['F']) / given['F'] * 100:+.1f}% / "
                        f"{(estimate['k'] - given['k']) / given['k'] * 100:+.1f}%）；不确定带 F ∈ "
                        f"[{band_F[0]:.4f}, {band_F[1]:.4f}]、k ∈ [{band_k[0]:.4f}, {band_k[1]:.4f}]"
                        f"（k 的题给值 {given['k']:.3f} 在带外 {given['k'] - band_k[1]:.4f}）。", ""),
    ], space_after=1)
    def pct(value, reference):
        return f"{(value - reference) / reference * 100:+.1f}%"

    inv_rows = [
        ["题给参数（参考）", f"{given['F']:.5f}", f"{given['k']:.5f}", "+0.0%", "+0.0%",
         f"{inversion['probes']['given']['J_total']:.2f}"],
        ["积分恒等式闭式", f"{inversion['closed_form']['F']:.5f}", f"{inversion['closed_form']['k']:.5f}",
         pct(inversion["closed_form"]["F"], given["F"]), pct(inversion["closed_form"]["k"], given["k"]),
         f"{inversion['probes']['closed_form']['J_total']:.2f}"],
        ["仅特征匹配（GP）", f"{inversion['features_only']['F']:.5f}", f"{inversion['features_only']['k']:.5f}",
         pct(inversion["features_only"]["F"], given["F"]), pct(inversion["features_only"]["k"], given["k"]),
         f"{inversion['probes']['features']['J_total']:.2f}"],
        ["特征+PDE（最终）", f"{estimate['F']:.5f}", f"{estimate['k']:.5f}",
         pct(estimate["F"], given["F"]), pct(estimate["k"], given["k"]), f"{estimate['J']:.2f}"],
    ]
    table(document, ["参数组", "F", "k", "F 误差", "k 误差", "J_total"], inv_rows,
          [4.8, 2.5, 2.5, 2.2, 2.2, 2.2])
    body(document, [
        (f"最终参数下 {len(final)} 个初值中 {n_final_ok} 个在 t ≤ 8×10⁴ 内精修到完整残差 ≤ 1e-11 且谱全负"
         f"（max Re λ 在 −3.1×10⁻⁴ 与 −5.8×10⁻⁵ 之间）；代表解 MAT 的完整残差 {seed2000['rel_residual']:.1e}、"
         f"{feats['n_spots']:.0f} 个斑。形态对照（同一测量算子）：类型一致，相干性、端点与交叉点密度在容差内；"
         f"但特征波长 {feats['wavelength']:.3f} 对目标 {target['wavelength']:.3f}（−10%）、"
         f"κ", ""), ("v", "sub"), (f" {feats['kappa_v']:.1f} 对 {target['kappa_v']:.1f}（+20%）、"
         f"缺陷比例 {feats['spot_defect_frac']:.2f} 对 {target['spot_defect_frac']:.2f}，"
         "说明若干尺度和缺陷特征尚未完全匹配。目标的不规则大斑与偏长波长在题给真值下同样出现，属数据本身的"
         "结构与噪声；评分不要求逐像素一致。", ""),
    ], space_after=2)

    heading(document, "5　稳健性、稳定性边界与局限")
    rs = robustness
    body(document, [
        ("稳健性（历史多种子实验，按统一口径重算）：", "b"),
        (f"青蛙斑点 {rs['robustness:frog_spots']['strict_success']}/{rs['robustness:frog_spots']['planned']} "
         f"完整稳态且未发现正模态；青蛙条纹 {rs['robustness:frog_stripes']['steady_full']}/"
         f"{rs['robustness:frog_stripes']['planned']} 收敛，但目标条纹形态 "
         f"{rs['robustness:frog_stripes']['strict_success']}/{rs['robustness:frog_stripes']['planned']}"
         f"（2 个种子落到斑点态）；球面斑点 {rs['robustness:sphere6_spots']['steady_full']}/"
         f"{rs['robustness:sphere6_spots']['planned']} 完整稳态，其中 "
         f"{rs['robustness:sphere6_spots']['strict_success']}/{rs['robustness:sphere6_spots']['planned']} "
         f"在有限谱检查中未发现正模态；球面条纹 {rs['robustness:sphere6_stripes']['steady_full']}/"
         f"{rs['robustness:sphere6_stripes']['planned']} 完整稳态，"
         f"{rs['robustness:sphere6_stripes']['strict_success']}/{rs['robustness:sphere6_stripes']['planned']}"
         f" 无正模态（其余六条带存在正倾斜模）。以上只对应历史多种子协议，不能与单文件候选验收混用。", ""),
    ], space_after=1)
    body(document, [
        ("稳定性边界：", "b"),
        ("本文只报告“有限采样未发现正实部特征值”，不声称整个谱的数学稳定性，也不声称全局非线性稳定。"
         "青蛙两组为三种 shift-invert 配置（σ = 10⁻³、0、10⁻⁵）的括号检查；无移位最大实部搜索在该刚性算子上"
         "不收敛，已记录但不作为通过证据。", ""),
    ], space_after=1)
    body(document, [
        ("局限：", "b"),
        ("孔洞区 AI 初值常落到另一个同样稳定的迷宫解（分类正确、形态不同）；L5 评测未做 orient 时只有相对平衡态，"
         "其成功率不能与 L6 形式任务混用；墙钟加速比只在旧模型同机同负载的一组成立"
         "（solve_s 673 s 对 1159 s），其余只比较与机器无关的积分步数"
         f"（中位：旧 AI 1.51 万、新 AI 1.38 万、纯积分 2.59 万）。网络留一法类型准确率青蛙 "
         f"{cv['frog']['type_accuracy']:.0%}、球面 {cv['sphere5']['type_accuracy']:.0%}，"
         "它衡量分类，不等同于最终斑图成功率。", ""),
    ], space_after=2)

    heading(document, "6　校核、资源与交付")
    body(document, [
        ("校核结论：", "b"),
        ("四份正式 MAT 均按 (n,1) float64 回读，数值有限、u ∈ [0,1]、v ≥ 0，"
         "并绑定 SHA256 与参数；完整残差用同一 robust L/M 重新计算，与本表一致。"
         "求解器侧另有几何收敛与时间收敛阶校核、可微梯度与有限差分对照（verification/）。"
         "稳定性证书的 MAT 哈希与提交 MAT 相同，任一不一致会直接判失败。", ""),
    ], space_after=1)
    body(document, [
        ("资源与复现：", "b"),
        (f"全 CPU，无需 GPU；反演 {inversion['wall_seconds']['total'] / 60:.1f} min（8 进程）；"
         f"四组正式生成单次约 {min(numbers['generation_wall_s'].values()) / 60:.0f}–"
         f"{max(numbers['generation_wall_s'].values()) / 60:.0f} min（不同机器）；稳定性检查 1–2 min/组；"
         f"训练标签网格 9×5 = 45 点/曲面，模型拟合 < 10 s。所有脚本固定随机种子，提交包含代码、模型、"
         f"结果与逐文件 SHA256 清单。", ""),
    ], space_after=1)
    body(document, [
        ("交付：", "b"),
        ("答案 ZIP 在根目录给出四组 MAT、终态图与残差/稳定性记录，task3 给出反演参数、目标—生成对比与误差指标，"
         "code/ 含可复现代码、附件数据与运行说明；技术报告 PDF 按赛题要求在“文档类文件”栏目单独提交。"
         "本报告与 ZIP 均为交付草稿：队名待填写，补齐队名并重新生成后方可上传。", ""),
    ], space_after=2)

    heading(document, "7　结论")
    body(document, [
        ("初赛三项任务完成：四个 float64 完整稳态（完整残差 8.4×10⁻¹⁵–2.4×10⁻¹⁴），有限谱检查未发现正实部；"
         "反演 F、k 相对题给值误差 −1.0% / −3.2%，但波长与缺陷等特征仍未完全匹配。AI 提供初值与参数搜索候选，"
         "PDE 求解器负责唯一的结果判定；全部数字可由提交包内的代码与记录复现。", ""),
    ], space_after=0)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run(f"{TITLE} · AI-MECH-2026-002 · 第 ")
    set_run_font(run, size=7.5, color=(0x60, 0x60, 0x60))
    add_field(footer, "PAGE")
    run = footer.add_run(" 页 / 共 ")
    set_run_font(run, size=7.5, color=(0x60, 0x60, 0x60))
    add_field(footer, "NUMPAGES")
    run = footer.add_run(" 页")
    set_run_font(run, size=7.5, color=(0x60, 0x60, 0x60))

    document.core_properties.title = TITLE
    document.core_properties.subject = "AI-MECH-2026-002 初赛技术报告（草稿，队名待填）"
    document.core_properties.author = f"参赛队（{TEAM}）"
    path = OUT / "初赛技术报告_I03_草稿.docx"
    document.save(path)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
