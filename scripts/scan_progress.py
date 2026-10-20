"""Progress of a running trajectory scan: one line per group from its log.

    python scripts/scan_progress.py                      # results/scan_stats
    python scripts/scan_progress.py --dir results/scan --watch 30
    python scripts/scan_progress.py --dir results/band --t-max 3000 --expect 25 --watch 30
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"t=\s*(\d+) alive=\s*(\d+)/(\d+).*?\((\d+)s\)")
STATS = re.compile(r"best shape_d=([\d.]+|inf).*?n\(shape_d<=1\)=(\d+)")
DONE = re.compile(r'"n_with_match": (\d+), "best_shape_d": ([\d.e+-]+|Infinity)')
SECS = re.compile(r"\((\d+)s\)\s*$")


def report(d: Path, t_max: float, expect: int = 0) -> str:
    rows, total_done, n_groups = [], 0, 0
    for log in sorted(d.glob("*.log")):
        n_groups += 1
        lines = [ln for ln in log.read_text().splitlines() if ln.startswith("[")]
        name = log.stem
        if not lines:
            rows.append(f"{name:24s} 启动中")
            continue
        last = lines[-1]
        if "DONE" in last:
            total_done += 1
            m = DONE.search(last)
            extra = f"匹配 {m.group(1):>3} 条  最佳距离 {float(m.group(2)):.2f}" if m else ""
            sm = SECS.search(last)
            secs = int(sm.group(1)) if sm else 0
            rows.append(f"{name:24s} 完成   {'#' * 20}  100%  用时 {secs // 60:3d} 分  {extra}")
            continue
        m = LINE.search(last)
        t, alive, B, secs = (int(x) for x in m.groups())
        frac = t / t_max
        eta = secs * (1 - frac) / max(frac, 1e-9) / 60
        s = STATS.search(last)
        extra = f"匹配 {s.group(2):>3} 条  最佳距离 {float(s.group(1)):.2f}" if s else ""
        bar = "#" * int(20 * frac) + "." * (20 - int(20 * frac))
        rows.append(f"{name:24s} t={t:5d}  {bar} {frac:4.0%}  存活 {alive:3d}/{B}  已用 {secs // 60:3d} 分  "
                    f"剩余≈{eta:3.0f} 分  {extra}")
    pattern = r"^[^ ]*python[^ ]* scripts/(trajectory_scan|band_dataset)\.py"
    running = len(subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split())
    if expect > n_groups:
        rows.append(f"（另有 {expect - n_groups} 组排队中）")
        n_groups = expect
    head = (f"{datetime.now():%H:%M:%S}  {d.resolve().relative_to(ROOT)}  完成 {total_done}/{n_groups} 组  "
            f"运行中进程 {running}")
    note = "（剩余时间按目前速度线性估计，偏保守：轨迹死亡或稳定后会加速）"
    return "\n".join([head, note, *rows]), total_done == n_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "scan_stats"))
    ap.add_argument("--t-max", type=float, default=4000)
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    ap.add_argument("--expect", type=int, default=0, help="total number of groups (some may not have started)")
    args = ap.parse_args()
    d = Path(args.dir)
    while True:
        out, finished = report(d, args.t_max, args.expect)
        if args.watch:
            print("\033[2J\033[H" + out + ("\n全部完成。" if finished else ""), flush=True)
            if finished:
                break
            time.sleep(args.watch)
        else:
            print(out)
            break


if __name__ == "__main__":
    main()
