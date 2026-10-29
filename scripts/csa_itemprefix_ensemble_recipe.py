"""Four-stage fixed matrix; keep scheduler and read-only analysis stdlib only."""

EXPERIMENT_REVISION = 'csa-itemprefix-ensemble-v2-resampled-T'
STAGES = {
    'S0': ('csa', 'item_prefix_joint', 'score_025', 'score_mean', 'score_075', 'score_degree'),
    'S1': ('gate_degree', 'gate_user', 'platt_mean', 'degree_only'),
    'H': ('hidden_concat', 'hidden_concat_degree', 'hidden_project', 'hidden_project_degree',
          'hidden_csa_only', 'hidden_prefix_only'),
    'U': ('shadow_concat', 'shadow_concat_degree', 'shadow_project', 'shadow_project_degree',
          'shadow_csa_only', 'shadow_prefix_only'),
}
# Standalone follow-up stages include unchanged expert references and raw mean.
SUITES = {'S0': STAGES['S0']}
for stage in ('S1', 'H', 'U'):
    SUITES[stage] = ('csa', 'item_prefix_joint', 'score_mean', *STAGES[stage])
SUITES['all'] = tuple(name for conditions in STAGES.values() for name in conditions)
OOF_CONDITIONS = ('gate_degree', 'gate_user', 'platt_mean')
TRAINING = dict(selection_metric='validation_all_auc', num_epochs=50, patience=5, batch_size=512, lr=.001, weight_decay=.0001,
                platt_max_iter=100, platt_l2=.0001)


def contrasts(conditions):
    pairs = [(name, reference) for name in conditions
             for reference in ('csa', 'item_prefix_joint', 'score_mean') if name != reference]
    pairs += [(name, control) for name in conditions for control in
              ('hidden_csa_only', 'hidden_prefix_only') if name.startswith('hidden_') and name != control]
    pairs += [(name, control) for name in conditions for control in
              ('shadow_csa_only', 'shadow_prefix_only') if name.startswith('shadow_') and name != control]
    return [pair for pair in dict.fromkeys(pairs) if all(name in conditions for name in pair)]


def source_fingerprint():
    """Pin worker implementation/configs, including upstream expert defaults."""
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    files = list((root / 'sralt').rglob('*.py')) + list((root / 'sralt/configs').rglob('*.yaml'))
    files += [Path(__file__), *(root / 'scripts').glob('*csa_itemprefix_ensemble*.py'),
              *(root / 'scripts').glob('*tail_proxy*.py'), root / 'scripts/item_prefix_case.py',
              root / 'scripts/item_prefix_recipe.py', root / 'scripts/item_prefix_csa_case.py',
              root / 'scripts/run_restored_baseline.py', root / 'scripts/experiment_progress.py']
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
