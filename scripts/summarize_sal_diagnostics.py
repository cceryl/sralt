#!/usr/bin/env python3
"""Extract structured SAL observations from existing logs; never load models."""
import argparse
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    records, coverage = [], []
    for case in manifest['cases']:
        if not case.get('sal_weight'):
            continue
        path = args.run_dir / case['output'] / 'console.log'
        observations = []
        if path.exists():
            for line in path.read_text().splitlines():
                if 'SAL_DIAGNOSTIC ' in line:
                    payload = line.split('SAL_DIAGNOSTIC ', 1)[1]
                    payload = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', payload)
                    record = json.loads(payload)
                    if record['kind'] == 'setup' and 'seed' in record:
                        record['diagnostic_seed'] = record.pop('seed')
                    observations.append(record)
        identity = {key: case[key] for key in ('target', 'dataset', 'seed', 'condition', 'sal_weight')}
        records.extend(identity | record for record in observations)
        phases = {r.get('phase') for r in observations}
        coverage.append(identity | {'status': case['status'], 'records': len(observations),
                                    'checkpoint': 'checkpoint' in phases,
                                    'refit': 'checkpoint_refit' in phases,
                                    'note': 'observed' if observations else 'missing; inspect cache hit or failure'})
    destination = args.run_dir / 'diagnostics'
    destination.mkdir(exist_ok=True)
    (destination / 'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    (destination / 'coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')
    for kind in ('proxy', 'gradient', 'refresh'):
        rows = [r for r in records if r['kind'] == kind]
        if not rows:
            continue
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (destination / f'{kind}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)
    budgets = {}
    for record in records:
        if record['kind'] != 'refresh':
            continue
        key = tuple(record[k] for k in ('target', 'dataset', 'seed', 'condition', 'sal_weight', 'epoch'))
        budget = budgets.setdefault(key, dict(refresh_count=0, fit_epochs=0, optimizer_steps=0))
        budget['refresh_count'] += 1
        budget['fit_epochs'] += record['fit_epochs']
        budget['optimizer_steps'] += record['optimizer_steps']
    if budgets:
        keys = ['target', 'dataset', 'seed', 'condition', 'sal_weight', 'epoch']
        with (destination / 'refresh_budget.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys + ['refresh_count', 'fit_epochs', 'optimizer_steps'])
            writer.writeheader()
            writer.writerows(dict(zip(keys, key)) | value for key, value in budgets.items())
    print(f'{len(records)} observations; {destination}; missing logs are listed in coverage.json')


if __name__ == '__main__':
    main()
