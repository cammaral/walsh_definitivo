#!/usr/bin/env python3
"""Compare regenerated summaries to the values in the September 2026 paper.

Numerical agreement is approximate, not bitwise reproducibility.  No data are
fabricated when a required stage output is missing.
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ALPHA_BETA = {
    (9, 4): (27, 0.75),
    (10, 5): (22, 0.75),
    (11, 5): (22, 0.75),
    (12, 6): (36, 0.75),
    (14, 7): (42, 0.50),
    (16, 8): (80, 0.50),
}
FULL_AR = {100: .9762, 200: .9759, 400: .9765, 500: .9752,
           800: .9826, 2000: .9792, 7000: .9860}
B1_AR = {100: .9770, 200: .9769, 400: .9787, 500: .9771,
         800: .9804, 2000: .9835, 7000: .9850}
GSET_AR = {
    'G14': (.9698, .9828),
    'G23': (.9874, .9926),
    'G60': (.9746, .9781),
    'G64': (.9195, .9369),
}
STATE_FIDELITY = {
    'G14': .999981, 'G23': .999949, 'G60': .999950, 'G64': .999953,
}

issues = []

def check(label, actual, expected, atol=0):
    passed = abs(float(actual) - float(expected)) <= atol
    print(f"{'OK  ' if passed else 'FAIL'} {label}: actual={actual}  expected={expected}  atol={atol}")
    if not passed:
        issues.append(label)

def read(relative):
    path = ROOT / relative
    if not path.is_file():
        issues.append(f'MISSING {relative}')
        print(f'FAIL missing: {relative}')
        return None
    return pd.read_csv(path)

def stage_01():
    df = read('data/graphs/graphs.csv')
    if df is not None:
        check('01 graphs count', len(df), 1750)
        for n in FULL_AR:
            check(f'01 graphs n={n}', sum(df['n'] == n), 250)
    ref = read('results/stage_01/mqlib_summary.csv')
    if ref is not None:
        check('01 MQLib reference count', len(ref), 1750)

def stage_02():
    split = read('data/graphs/split.csv')
    if split is not None:
        check('02 calibration count', sum(split['split'] == 'calibration'), 350)
        check('02 held-out count', sum(split['split'] == 'evaluation'), 1400)
    alpha = read('results/stage_02/selected_alpha.csv')
    beta = read('results/stage_02/selected_beta.csv')
    if alpha is None or beta is None:
        return
    for (q, k), (a, b) in ALPHA_BETA.items():
        ax = alpha[(alpha.q == q) & (alpha.k == k)]
        bx = beta[(beta.q == q) & (beta.k == k)]
        if len(ax) != 1 or len(bx) != 1:
            issues.append(f'02 missing or duplicated config {(q,k)}')
            continue
        check(f'02 alpha {(q,k)}', ax.iloc[0]['alpha'], a)
        check(f'02 beta {(q,k)}', bx.iloc[0]['beta'], b)

def stage_03():
    df = read('results/stage_03/summary.csv')
    if df is None:
        return
    for n, ar in FULL_AR.items():
        row = df[df.n == n]
        if len(row) != 1:
            issues.append(f'03 size {n} missing/duplicated')
            continue
        check(f'03 n={n} mean AR', row.iloc[0]['mean_AR'], ar, 0.00015)

def stage_04():
    df = read('results/stage_04/summary.csv')
    if df is None:
        return
    for n, ar in B1_AR.items():
        row = df[(df.n == n) & (df.B == 1)]
        if len(row) != 1:
            issues.append(f'04 size {n} B=1 missing/duplicated')
            continue
        check(f'04 n={n} B1 mean AR', row.iloc[0]['mean_AR'], ar, .00015)

def stage_05():
    df = read('results/stage_05/summary.csv')
    if df is None:
        return
    for g, (raw, refined) in GSET_AR.items():
        row = df[df.instance == g]
        if len(row) != 1:
            issues.append(f'05 instance {g} missing/duplicated')
            continue
        row = row.iloc[0]
        check(f'05 {g} B', row['B'], 1)
        check(f'05 {g} raw AR', row['mean_raw_AR'], raw, .00015)
        check(f'05 {g} refined AR', row['mean_local_AR'], refined, .00015)

def stage_06():
    df = read('results/stage_06/05_summary/statevector_summary.csv')
    if df is None:
        return
    for g, fidelity in STATE_FIDELITY.items():
        row = df[df.instance == g]
        if len(row) != 1:
            issues.append(f'06 instance {g} missing/duplicated')
            continue
        row = row.iloc[0]
        check(f'06 {g} fidelity', row['state_fidelity_target'], fidelity, 0.000005)
        check(f'06 {g} strict epsilon', row['er'], .02, 1e-10)
        check(f'06 {g} sign agreement', row['correlator_sign_agreement'], 1., 1e-10)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--through', choices=('02', '03', '04', '05', '06'), default='06')
    args = p.parse_args()
    stages = [stage_01, stage_02, stage_03, stage_04, stage_05, stage_06]
    n = int(args.through)
    for fn in stages[:n]:
        print(f'\n## {fn.__name__}')
        fn()
    if issues:
        print(f'\nFAILED: {len(issues)} checks; first problems: {issues[:15]}')
        raise SystemExit(1)
    print('\nPASS: available stage results are consistent with the paper within printed-value tolerance.')

if __name__ == '__main__':
    main()
