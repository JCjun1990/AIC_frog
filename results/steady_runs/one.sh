#!/bin/zsh
# One steady-state job; arguments are passed straight to run_baseline.py (must include --name).
cd "$(dirname "$0")/../.."
name=""; prev=""
for a in "$@"; do [[ "$prev" == "--name" ]] && name="$a"; prev="$a"; done
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_baseline.py --surface frog --pattern spots \
  --t-max 100000 --out results/steady_runs "$@" > "results/steady_runs/${name}.log" 2>&1
