#!/usr/bin/env bash
# Copy the code and the two data files from this Mac to the RTX 4080 laptop (ssh alias rtx4080, see
# README "Windows RTX 4080 laptop") into C:\Users\18650\turing. Run it again after every code change.
# results/ stays on the Mac; fetch laptop outputs with  scp -r rtx4080:turing/results/<dir> results/
set -euo pipefail
cd "$(dirname "$0")/../.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
COPYFILE_DISABLE=1 tar --no-xattrs --exclude=__pycache__ --exclude=.DS_Store -czf "$tmp/code.tgz" \
    turing_rd scripts requirements.txt README.md frog.mat Solution.mat
ssh rtx4080 "if not exist turing mkdir turing" 2>/dev/null
scp -q "$tmp/code.tgz" rtx4080:turing/code.tgz
ssh rtx4080 "cd turing && tar -xzf code.tgz && del code.tgz" 2>/dev/null
echo "pushed $(du -h "$tmp/code.tgz" | cut -f1) to rtx4080:C:\\Users\\18650\\turing"
