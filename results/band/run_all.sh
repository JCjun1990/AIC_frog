#!/bin/zsh
# Generate all band-dataset groups, 8 at a time. Usage: zsh results/band/run_all.sh
cd "$(dirname "$0")/../.."
cat results/band/jobs.txt | xargs -P 8 -L 1 zsh -c 'VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/band_dataset.py --group $0 --du-scale $1 --ratio $2 > results/band/g$(printf %02d $0).log 2>&1'
echo ALL_DONE
