"""Two-page scientific report from saved evidence and an executed-test record.

python scripts/report_v3.py --out <new.pdf> --font <CJK-font.ttf> --test-results <tests.json>
The original PDFs, numerical results, and ZIPs are never overwritten.
"""
from pathlib import Path
import argparse
import re
from datetime import datetime, timezone, timedelta
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import evidence_stats as E

SUB = ROOT / "submission"
TASKS = (("task1_sphere/spots", "球面斑点"), ("task1_sphere/stripes", "球面条纹"),
         ("task2_frog/spots", "青蛙斑点"), ("task2_frog/stripes", "青蛙条纹"))


def wrap_lines(font, value, size, max_width):
    """Preserve decimals, exponents and fractions when wrapping report text."""
    output_lines = []
    for paragraph in str(value).splitlines() or [""]:
        line = ""
        tokens = re.findall(r"[+-]?[A-Za-z0-9_]+(?:[./:+%-][A-Za-z0-9_]+)*%?|.", paragraph)
        for token in tokens:
            if line and font.text_length(line+token, fontsize=size)>max_width:
                output_lines.append(line); line=token
            else: line+=token
        output_lines.append(line)
    return output_lines


def report_sections(c, tests_run):
    """Pure, testable science statements shared with Markdown generators."""
    sections = {**E.claims(c),
            "verification": f"本轮实际执行 {tests_run} 项单元测试并通过，数量来自运行记录，不是源码计数。"
                            "四组正式结果的完整残差及三配置谱证书独立验收。有限谱检查不等于全谱数学证明。",
            "methods": "Gray-Scott：∂u/∂t = Du Δ_s u − uv² + F(1−u)；∂v/∂t = Dv Δ_s v + uv² − (F+k)v。"
                       "[D]=L²/T，[F]=[k]=1/T，u、v 无量纲。采用 robust Laplace-Beltrami 离散与集中质量矩阵 M，"
                       "IMEX SBDF2 时间推进，Newton 精修。完整稳态残差 R=√((RMS_M(Ru)²+RMS_M(Rv)²)/2)/F ≤ 1e-11，"
                       "RMS_M(a)=√(ΣMa²/ΣM)，Ru、Rv 是实际离散 PDE 双变量右端；不做转动投影。",
            "ai": "AI 采用小型 MLP 集成，由 (F,k) 预测类型及斑图统计，再构造初值；PDE/Newton 求解并校核。"
                  "训练来源为自生成数值终态，含未达完整稳态的标签。正式模型 v1 与研究改进模型 v2 分开保留。",
            "limits": "历史多种子及新增条带实验沿用各自协议，未在本轮从头重跑。训练范围内测试不能作为范围外泛化证据。"
                      "特征对残差是后向误差，不是严格前向特征值误差界；未证明全局非线性稳定性。"
                      "不同初值终态标签差异不能单独证明动力学多稳态。任务 4 属总决赛，本报告不作实现声明。"}
    if c.get("methodology"):
        inv = c["methodology"]["inversion"]; counts = inv["counts_from_aggregate_only"]
        sections["limits"] += (f"历史反演 {inv['total']} 条终态，polished {counts['polished']}、旧规则接受 {counts['legacy_accepted']}；"
            "逐条场缓存缺失，严格稳态数和参数偏移未知。未来参考库使用完整双变量残差，旧协议隔离保留。")
    return sections


def render_report(c, sub, *, font_path, output, tests_run, team="待填", date="", best_match=None):
    """Render saved records only. No training, field generation, or eigensolve."""
    sub, output, font_path = Path(sub), Path(output), Path(font_path)
    if output.exists():
        raise FileExistsError("report exists; nothing overwritten")
    best_match = sub / "task3_inversion/best_match" if best_match is None else Path(best_match)
    font = fitz.Font(fontfile=str(font_path))
    sections = report_sections(c, tests_run)
    required = "∂Δ−²≤κ₀σ√Σ∫∈×" + "".join(sections.values())
    missing = sorted({ch for ch in required if not ch.isspace() and not font.has_glyph(ord(ch))})
    if missing:
        raise ValueError("font lacks required scientific glyphs: " + "".join(missing))
    doc = fitz.open(); margin, width = 34., 595.
    ink, blue, grey = (.12,.16,.20), (.10,.27,.43), (.40,.44,.48)
    def page(number):
        p = doc.new_page(width=width, height=842.)
        p.insert_font(fontname="cn", fontfile=str(font_path))
        p.draw_line((margin,810),(width-margin,810),color=(.7,.75,.8))
        p.insert_text((margin,823),f"AI-MECH-2026-002 · {team} · 统计修订 · {number}/2",fontname="cn",fontsize=8,color=blue)
        return p
    def lines(value, size, max_width):
        return wrap_lines(font, value, size, max_width)
    def text(p, value, y, size=8.7, step=12.1, x=margin, max_width=width-2*margin, color=ink):
        for line in lines(value,size,max_width):
            if y>798: raise ValueError(f"report overflow at y={y}")
            p.insert_text((x,y),line,fontname="cn",fontsize=size,color=color); y+=step
        return y
    def heading(p, value, y):
        return text(p,value,y,size=11,step=16,color=blue)
    def table(p, rows, columns, y, size=8.0):
        edges=columns+[width-margin]
        for row in rows:
            cells=[lines(value,size,edges[i+1]-edges[i]-7) for i,value in enumerate(row)]
            height=max(len(cell) for cell in cells)*11.5+4
            for i,cell in enumerate(cells):
                text(p,"\n".join(cell),y,size=size,step=11.5,x=columns[i],max_width=edges[i+1]-edges[i]-5)
            y+=height
        return y
    p=page(1)
    y=text(p,"曲面图灵斑图的 AI-PDE 生成与反演",52,size=16,step=22)
    y=text(p,f"队名：{team} · {date} · 任务 1/2/3 · 数值终态与完整稳态分层",y,size=8.3,color=grey)
    y=heading(p,"1  方法与稳态判据",y+7)
    y=text(p,sections["methods"],y)
    y=text(p,sections["ai"],y+2)
    y=heading(p,"2  四组正式结果：完整残差与有限谱证据",y+6)
    rows=[("曲面/形态","数量","波长","完整残差 R","最大已采样 Re λ","MAT SHA256")]
    for task,label in TASKS:
        s=E.load(sub/task/"summary.json"); cert=E.load(sub/task/"stability_certificate.json")
        shape=s.get("pattern_after") or s.get("pattern") or {}; decision=cert["decision"]
        mx=decision.get("full_max_real_checked")
        if mx is None: mx=max(decision["sampled_maxima"])
        rows.append((label,str(shape["n_components"]),f"{shape['wavelength']:.3f}",
                     f"{s['mat_certificate']['residual']['value']:.3e}",f"{mx:+.3e}",E.sha(sub/task/"solution.mat")[:10]))
    y=table(p,rows,[margin,108,145,194,293,427],y+2)
    y=text(p,"完整 Jacobian 未排除转动模态。球面为两移位加直接 LR；青蛙为三移位，直接 LR 未收敛的历史尝试不计通过。",y+2)
    y=heading(p,"3  AI 泛化与积分步数：指标范围分开",y+5)
    for key in ("generalization","efficiency","bands"):
        y=text(p,sections[key],y+2,size=8.4,step=11.6)
    top=y+3; row_h=min(142.,(800-top)/2)
    if row_h<70: raise ValueError("insufficient space for formal result images")
    for i,(task,label) in enumerate(TASKS):
        x=margin+(i%2)*265; image_y=top+(i//2)*row_h
        text(p,label,image_y+8,size=8,x=x,max_width=250)
        p.insert_image(fitz.Rect(x,image_y+11,x+258,image_y+row_h-2),filename=str(sub/task/"final_v.png"),keep_proportion=True)
    p=page(2); inv,b=c["inversion"],c["best_match"]
    y=heading(p,"4  任务 3：参数估计与固定参数多初值匹配",50)
    e,cf=inv["estimate"],inv["closed_form"]
    y=text(p,"预处理用热核低通并截断至物理范围。闭曲面积分扩散项为零，得到 F=∫uv²/∫(1−u)，F+k=∫uv²/∫v，"
             "作为初值；随后采用特征与积分判据、共同随机初值及 GP 代理估计参数。",y)
    given=inv["given_spots"]
    y=table(p,[("记录","F","k","相对参考误差"),
               ("正式估计",f"{e['F']:.7f}",f"{e['k']:.7f}",f"{e['F']/given['F']-1:+.2%} / {e['k']/given['k']-1:+.2%}"),
               ("积分初值",f"{cf['F']:.7f}",f"{cf['k']:.7f}","启发式初始估计")],
            [margin,150,250,350],y+3)
    y=text(p,sections["inverse"],y+3)
    target,old,best=b["target"],b["previous_representative"],b["best"]
    y=table(p,[("同一特征测量","目标","最佳匹配","旧代表解"),
               ("斑点数",target['n_final'],best['n_final'],old['n_final']),
               ("特征距离 J","0",f"{best['J']:.3f}",f"{old['J']:.3f}"),
               ("谱峰波长",f"{target['peak_wavelength']:.3f}",f"{best['peak_wavelength']:.3f}",f"{old['peak_wavelength']:.3f}"),
               ("最佳匹配完整残差","—",f"{b['mat_certificate']['residual']['value']:.3e}","历史代表解另存")],
            [margin,194,300,422],y+5)
    p.insert_image(fitz.Rect(margin,y+2,width-margin,y+147),filename=str(best_match/"report_panel.png"),keep_proportion=True)
    y+=156
    y=heading(p,"5  验证、资源与局限",y+3)
    y=text(p,sections["verification"],y)
    y=text(p,"纯 CPU 历史计算记录：反演总时长 "f"{inv['wall_seconds']['total']/60:.1f} 分钟，最佳匹配搜索 {b['wall_seconds']/60:.1f} 分钟。"
             "这些是各自环境的历史耗时，不能当作本轮墙钟加速对照。",y+2)
    y=text(p,sections["limits"],y+2)
    y=text(p,"复现统计：python scripts/evidence_stats.py --out <新目录>；运行测试：python scripts/run_test_evidence.py --out <新记录>。"
             "正式 MAT 验收：python code/scripts/submission_workflow.py verify --submission .",y+4,size=8.0,step=11.5)
    y=text(p,"参考：Pearson, Science 261 (1993)；Sharp & Crane, SGP (2020)；Kazarnikov et al., arXiv:2504.02530 (2025)；"
             "Schnörr & Schnörr, Machine Learning 112 (2023)。方法贡献范围以本报告保存证据为准。",y+3,size=7.7,step=11,color=grey)
    doc.subset_fonts(); doc.save(output,garbage=4,deflate=True);doc.close()
    return {"pages":2,"page2_bottom":y,"tests_run":tests_run,"font":str(font_path)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--font",type=Path,required=True)
    ap.add_argument("--test-results",type=Path,required=True)
    ap.add_argument("--quality",type=Path,default=SUB/"evidence/generalization/QUALITY_BREAKDOWN.json")
    ap.add_argument("--team",default="待填")
    ap.add_argument("--date",default=datetime.now(timezone(timedelta(hours=8))).date().isoformat())
    ap.add_argument("--best-match",type=Path,default=SUB/"task3_inversion/best_match")
    args=ap.parse_args()
    n=E.validate_test_record(E.load(args.test_results),ROOT)
    q=E.load(args.quality)
    for rel,expected in q["provenance"]["files"].items():
        if E.sha(ROOT/rel)!=expected: raise ValueError("statistics source changed: "+rel)
    print(render_report(E.context(ROOT,q),SUB,font_path=args.font,output=args.out,tests_run=n,
                        team=args.team,date=args.date,best_match=args.best_match))


if __name__=="__main__":
    main()
