#!/bin/zsh
# Steady states at the steady-hypothesis parameter sets, 8 at a time. Usage: zsh results/steady_runs/run_all.sh
cd "$(dirname "$0")/../.."
xargs -P 8 -L 1 zsh results/steady_runs/one.sh < results/steady_runs/jobs.txt
echo ALL_DONE
