#!/usr/bin/env python3
"""Run the complete extended reproducibility workflow."""
import argparse, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run(script,workers):
    cmd=[sys.executable,str(HERE/script)]
    if script not in {'analyze_extended.py','make_figures_extended.py','verify_reproducibility_extended.py'}:
        cmd += ['--workers',str(workers)]
    print('\n>>>',' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=HERE,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=8); args=ap.parse_args()
    for script in [
        'benchmark_extended.py',
        'tune_geometry_straddle.py',
        'tune_evidence_weights.py',
        'kernel_sensitivity.py',
        'robustness_ablation_extended.py',
        'analyze_extended.py',
        'make_figures_extended.py',
        'verify_reproducibility_extended.py',
    ]:
        run(script,args.workers)
if __name__=='__main__': main()
