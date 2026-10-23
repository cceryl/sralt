"""Small JSON artifact shared by the normal and low-memory pipelines."""

import json
from pathlib import Path


def save_attack_report(data, model, output_dir):
    report = {'groups': getattr(data, 'subgroup_metrics', {}),
              'shadow': getattr(data, 'shadow_diagnostics', {}),
              'normalization': getattr(model, 'normalization_diagnostics', {}),
              'sparse_training': getattr(model, 'training_diagnostics', [])}
    if hasattr(model, 'statistical_diagnostics'):
        report['statistical'] = model.statistical_diagnostics
    Path(output_dir, 'attack_diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
