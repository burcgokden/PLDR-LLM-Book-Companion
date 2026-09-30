#!/usr/bin/env python3
"""Unified book-companion checks; all outputs stay in this checkout."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'companions/dynamics'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check',choices=['integrity','scientific','resources','lean','evidence','standalone'])
    args,extra=parser.parse_known_args()
    python=sys.executable
    jobs={
      'integrity':[(ROOT,[python,'-B','scripts/verify_release.py']),
                   (ROOT,[python,'-B','scripts/check_documented_commands.py']),
                   (D,[python,'-B','scripts/verify_scientific_manifest.py']),
                   (D,[python,'-B','scripts/check_formal_manifest.py'])],
      'scientific':[(ROOT,[python,'-B','-m','pytest','-q','-p','no:cacheprovider','tests','companions/foundations/audit','--junitxml=validation/book-scientific.xml']),
                    (D,[python,'-B','scripts/run_scientific_checks.py'])],
      'resources':[(D,[python,'-B','scripts/'+name+'.py']) for name in
                   ['check_process_boundaries','check_campaign_budget','check_resource_execution']],
      'lean':[(ROOT,[python,'-B','scripts/check_lean.py',*extra])],
      'evidence':[(D,[python,'-B','scripts/verify_evidence.py',*extra])],
      'standalone':[(ROOT,[python,'-B','scripts/check_standalone.py',*extra])],
    }
    if extra and args.check in {'integrity','scientific','resources'}:parser.error('unexpected extra arguments')
    (ROOT/'validation').mkdir(exist_ok=True)
    for cwd,cmd in jobs[args.check]:
        result=subprocess.run(cmd,cwd=cwd)
        if result.returncode:return result.returncode
    return 0

if __name__=='__main__':raise SystemExit(main())
