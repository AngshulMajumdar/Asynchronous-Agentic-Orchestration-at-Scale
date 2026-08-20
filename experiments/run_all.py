#!/usr/bin/env python3
"""Repository reproduction entry point.

Default: rebuild every manuscript figure from committed fixed-seed raw outputs.
Use --fresh-revision to rerun the expensive revision experiments that produced
Figs. 42--49 and the gamma queue checks on the current CPU.
"""
from pathlib import Path
import argparse, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]

def run(path):
    print(f'\n==> {path.relative_to(ROOT)}')
    subprocess.run([sys.executable,str(path)],cwd=ROOT,check=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--fresh-revision',action='store_true',help='rerun revision experiments in addition to exact-data replot')
    args=ap.parse_args()
    run(ROOT/'experiments'/'replot_reference.py')
    if args.fresh_revision:
        run(ROOT/'experiments'/'archive'/'round1_revision_experiments.py')
        run(ROOT/'experiments'/'archive'/'round4_experiments.py')
    print('\nDone.')
