"""Generate a new evidence-aligned two-page draft; never overwrite the historical PDF."""
from pathlib import Path
import argparse, json, hashlib
import fitz

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
TASKS = (("task1_sphere/spots", "球面斑点", "45 斑"),
         ("task1_sphere/stripes", "球面条纹", "7 条带 / seed2"),
         ("task2_frog/spots", "青蛙斑点", "22 斑"),
         ("task2_frog/stripes", "青蛙条纹", "5 段"))


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True, help="CJK TrueType font; embedded and subset")
    args = parser.parse_args()
    if args.out.exists(): parser.error("output exists; nothing overwritten")
    font = fitz.Font(fontfile=str(args.font)); doc = fitz.open()
    width, height, margin = 595., 842., 36.
    ink, blue = (0.12, .16, .20), (.10, .27, .43)

    def page(number):
        p = doc.new_page(width=width, height=height)
        p.insert_font(fontname="cn", fontfile=str(args.font))
        p.draw_line((margin, 805), (width-margin, 805), color=(.7, .75, .8))
        p.insert_text((margin, 820), f"AI-MECH-2026-002 | 交付草稿 | 队名：待填 | {number}/2", fontname="cn", fontsize=8, color=blue)
        return p

    def text(p, value, y, size=9.2, step=14., x=margin, max_width=width-2*margin):
        for para in value.split("\n"):
            line = ""
            for ch in para:
                if font.text_length(line+ch, fontsize=size) > max_width:
                    p.insert_text((x, y), line, fontname="cn", fontsize=size, color=ink)
                    y += step; line = ch
                else: line += ch
            p.insert_text((x, y), line, fontname="cn", fontsize=size, color=ink); y += step
        return y

    def heading(p, value, y):
        p.draw_rect(fitz.Rect(margin, y-13, width-margin, y+6), fill=(.92,.95,.98), color=None)
        text(p, value, y, size=10.5); return y+24

    p=page(1)
    y=text(p,"AI-PDE 曲面图灵斑图：初赛技术报告",55,size=17,step=24)
    y=text(p,"交付草稿 v2 · 2026-10-22 · 队名待填；当前文件不能作为已命名正式版上传。",y+2,size=9)
    y=text(p,"任务1、2、3；任务4仅总决赛。本稿对齐当前正式 MAT 与 I01/I02/I03 证据，保留失败和有限谱边界。",y+2)
    y=heading(p,"1  四组正式生成结果（均为实际 float64 MAT）",y+10)
    headers=["曲面 / 形态","数量","完整残差 R","最大已采样 Re λ","MAT SHA256 前缀"]
    xs=[margin,139,222,324,435]; row_h=33
    p.draw_rect(fitz.Rect(margin,y-13,width-margin,y+15),fill=(.87,.91,.95),color=None)
    for x,label in zip(xs,headers):text(p,label,y,size=8.4,x=x,max_width=120)
    y+=row_h
    for task,label,count in TASKS:
        directory=SUB/task; summary=load(directory/"summary.json"); c=load(directory/"stability_certificate.json")
        maximum=c["decision"].get("full_max_real_checked")
        if maximum is None: maximum=max(c["decision"]["sampled_maxima"])
        values=[label,count,f'{summary["mat_certificate"]["residual"]["value"]:.3e}',f'{maximum:+.3e}',sha(directory/"solution.mat")[:12]]
        for x,value in zip(xs,values):text(p,value,y,size=8.5,x=x,max_width=120)
        p.draw_line((margin,y+13),(width-margin,y+13),color=(.8,.84,.87));y+=row_h
    y=text(p,"R = sqrt((RMS_M(Ru)^2 + RMS_M(Rv)^2)/2)/F，RMS_M(a)^2 = sum(M a^2)/sum(M)。\n实际 robust L/M；不扣除转动；四组 R <= 1e-11。I01 证书绑定回读文件 SHA、参数、种子和算子。",y+2,size=8.7,step=13)
    y=text(p,"球面：两种 shift-invert + 直接 LR；青蛙：三种 shift-invert（0、1e-5、1e-3），每组40个特征对。\n全部已采样模态为负且特征对残差通过。青蛙直接 LR 不收敛，失败保留，不用于通过判定。\n有限谱采样不是数学全谱证明，也不证明全局非线性稳定。历史六条带 +1.665e-6 模态未被忽略。",y+3,size=8.7,step=13)
    image_y=y+8
    for i,(task,label,count) in enumerate(TASKS):
        x=margin+(i%2)*267; top=image_y+(i//2)*128
        text(p,f"{label} · {count}",top,size=9,x=x,max_width=250)
        p.insert_image(fitz.Rect(x,top+6,x+255,top+110),filename=str(SUB/task/"final_v.png"),keep_proportion=True)
    text(p,"图：由正式 MAT 渲染；球面条纹为七条带 seed2。旧六条带及旧报告仅作历史对照。",image_y+260,size=8.5)
    p=page(2);y=55
    y=heading(p,"2  数值模型、AI 参与与复现",y)
    y=text(p,"Gray-Scott：du/dt = Du ΔΓu - uv^2 + F(1-u)，dv/dt = Dv ΔΓv + uv^2 - (F+k)v。\nDu=2.78e-4、Dv=1.39e-4；u/v 无量纲，[D]=L^2/T，[F]=[k]=1/T。\nrobust 内蕴 Laplace-Beltrami 离散与集中质量，IMEX SBDF2、Newton 精修。球面先处理相对平衡族，最终仍以未投影完整残差验收；相对平衡态不等于完整离散稳态。\n小型 MLP 从 (F,k) 预测类型和统计量，再合成初值；所有输出由 PDE 复算。只用组委会 frog.mat、Solution.mat 和自生成训练数据，无外部生成模型。",y)
    y=heading(p,"3  历史泛化与稳健性：与正式文件验收分开",y+8)
    y=text(p,"Legacy Evaluation：2曲面×2流程×8参数×3种子，共96条历史记录。frog 旧判据 AI 24/24、普通22/24；sphere5 旧判据12/24、6/24，使用转动相对残差和排除转动模态，不能称完整稳态成功率。\n旧9×5网格留一类型分类：frog 29/45（64%）、sphere 40/45（89%）；不同密度的新模型 LOO 不能直接视为公平提升。不同机器或负载不宣称墙钟加速。\n历史 I04 审计摘要：frog斑点31/33，条纹完整稳态33/33、目标条纹31/33；sphere斑点完整稳态17/23、无正模态14/23；sphere条纹完整稳态22/23、无正模态9/23。精简 main 不含逐样本 I04 全量记录，以上标记为归档审计结果。\nadaptive v1/v2 未超过既有 A2；没有把未验证研究算法用于正式提交。",y)
    y=heading(p,"4  参数反演及尚存偏差",y+8)
    inv=load(SUB/"task3_inversion/summary.json");e=inv["estimate"]
    y=text(p,f"只反演 F、k，Du/Dv 固定。热核去噪与物理截断、积分恒等式初值、特征匹配、GP代理及多种子精修。\n最终 F={e['F']:.5f}、k={e['k']:.5f}；相对参考0.028/0.062，误差约 -1.0%/-3.2%。\n目标函数容差带 F [0.0267,0.0287]、k [0.0598,0.0613] 是启发式敏感性带，不是覆盖率校准置信区间；参考 k 在带外，系统偏低尚未消除。\n6个初值中5个精修成功；代表seed2000残差约8.7e-14、21斑。历史单配置谱不升级为三配置全谱证据。波长约0.491对目标0.547，缺陷比例0.29对0.60，尺度与缺陷尚未完全匹配。",y)
    y=heading(p,"5  资源、证据与交付限制",y+8)
    y=text(p,"CPU 计算，无需GPU。历史全量训练/反演的成本受缓存与硬件影响，本轮工程修复未重训或改变正式结果。\n包内 code/scripts/submission_workflow.py verify --submission <解压根目录> 可重算四组完整残差；prepare-models 可准备网络目录。单 shift 重算命令独立报告，不替换三配置证书。\n当前 ZIP/PDF 均为队名待填的草稿；PDF按赛题在文档栏目单独提交。昂贵的从头训练与全量生成本轮未复跑。完整性以 PACKAGE_MANIFEST.json 与独立解包验收为准。",y)
    if y>790:raise RuntimeError(f"Report page overflow at {y}")
    doc.set_metadata({"title":"AI-MECH-2026-002 初赛技术报告 交付草稿 v2","author":"队名待填","subject":"Evidence-aligned draft; finite numerical spectrum only"})
    doc.subset_fonts()
    args.out.parent.mkdir(parents=True,exist_ok=True);doc.save(args.out,garbage=4,deflate=True);doc.close()
    print(args.out)
    return 0

if __name__=="__main__":raise SystemExit(main())
