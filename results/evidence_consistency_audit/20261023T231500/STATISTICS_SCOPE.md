# 历史统计口径说明

此目录保留旧 JSON/CSV 的原始字节。旧重复数统计混用了双曲面总数与单曲面唯一键；旧 spec_checked 字段也未区分尝试与验证通过，不能作为当前质量统计。

维护入口为 scripts/evidence_stats.py；本目录 gen_stats.py 仅委托该入口并要求新输出目录。当前统计见 submission/evidence/generalization/QUALITY_BREAKDOWN.json 与同名 Markdown。
