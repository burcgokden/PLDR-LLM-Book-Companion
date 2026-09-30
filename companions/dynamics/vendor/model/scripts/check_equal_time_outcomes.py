#!/usr/bin/env python3
"""Fail closed on incomplete protocol-to-analysis-to-table outcome coverage."""
import argparse
from pathlib import Path
from numerical_validation import load_json_strict
from model_rg.provenance import sha256, write_json
from equal_time_outcomes import validate_outcomes

def check(analysis,protocol,rendered=None):
    a=load_json_strict(analysis.read_text());p=load_json_strict(protocol.read_text());v=validate_outcomes(a,p)
    if rendered is not None:raise ValueError('Only scientific JSON inputs are supported')
    return dict(v,analysis_sha256=sha256(analysis),protocol_sha256=sha256(protocol),checker_sha256=sha256(__file__),rendered_table_checked=rendered is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['analysis','protocol','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--rendered',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,check(a.analysis,a.protocol,a.rendered))
