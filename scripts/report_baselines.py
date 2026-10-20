"""Collect results/baseline/*: redraw convergence plots, build an overview figure and a summary table.

Writes results/baseline/overview.png and results/baseline/summary.md.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turing_rd.viz import plot_history  # noqa: E402

ORDER = ["sphere_spots", "sphere_stripes", "frog_spots", "frog_stripes"]


def load_rows(path: Path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if k not in ("phase", "type") and v not in ("", None) else v)
                         for k, v in r.items()})
    return rows


def main():
    base = ROOT / "results" / "baseline"
    lines = ["| 曲面 · 参数 | 类型 | 斑块数 | 波长 λ | t 终 | 时间推进收敛 | 精修后残差 / F | max Re λ(J) | 墙钟 (s) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    done = []
    for name in ORDER:
        d = base / name
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        rows = load_rows(d / "log.csv")
        pol = s.get("polish")
        plot_history(rows, d / "convergence.png", title=f"{name} convergence",
                     polish=pol["history"] if pol and pol.get("accepted") else None)
        fp, st = s["final_pattern"], s.get("linear_stability") or {}
        lines.append(f"| {name.replace('_', ' · ')} | {fp['type']} | {fp.get('n_components', 0)} | "
                     f"{fp['wavelength']:.3f} | {s['time']['t_final']:.0f} | "
                     f"{'是' if s['converged_time_integration'] else '否'} | "
                     f"{s['final_residual']['steady_rel']:.1e} | {st.get('max_real', float('nan')):.1e} | "
                     f"{s['time']['wall_seconds_total']:.0f} |")
        done.append(name)
    (base / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if done:
        fig, axes = plt.subplots(len(done), 1, figsize=(12, 3.3 * len(done)))
        axes = [axes] if len(done) == 1 else axes
        for ax, name in zip(axes, done):
            ax.imshow(mpimg.imread(base / name / "field_v.png"))
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(base / "overview.png", dpi=110)


if __name__ == "__main__":
    main()
