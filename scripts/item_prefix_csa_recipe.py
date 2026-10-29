"""ItemPrefixCSA experiment identities and one-factor ablations (stdlib only)."""

EXPERIMENT_REVISION = 'item-prefix-csa-experiment-prefix10-v2'
OVERRIDES = {
    'full': (),
    'no_user_graph': ('shadow.model.attribute.user_augmentation=false',),
    'no_prefix_graph': ('shadow.model.attribute.prefix_augmentation=false',),
    'no_semantic_graphs': ('shadow.model.attribute.user_augmentation=false',
                           'shadow.model.attribute.prefix_augmentation=false'),
    'no_user_matching': ('attack.model.attribute.user_matching=false',),
    'no_prefix_matching': ('attack.model.attribute.prefix_matching=false',),
    'no_matching': ('attack.model.attribute.user_matching=false', 'attack.model.attribute.prefix_matching=false'),
    'prefix_only': ('shadow.model.attribute.graph_mode=prefix',),
    'user_only': ('shadow.model.attribute.graph_mode=user',),
    'joint_graph': ('shadow.model.attribute.graph_mode=joint',),
    'mean_fusion': ('shadow.model.attribute.fusion=mean',),
    'gate_fusion': ('shadow.model.attribute.fusion=gate',),
    'random_user_graph': ('shadow.model.attribute.random_user_graph=true',),
    'fixed_user_codes': ('shadow.model.attribute.fixed_user_codes=true',),
    'no_sparse': ('attack.model.attribute.sparse_training=false',),
    'score_user': ('shadow.model.attribute.quantized_scoring=user',),
    'score_item': ('shadow.model.attribute.quantized_scoring=item',),
    'score_continuous': ('shadow.model.attribute.quantized_scoring=none',),
    'offline_rvq': ('shadow.model.attribute.rvq_training=offline', 'shadow.model.attribute.quantized_scoring=none'),
}
REFERENCES = ('baseline', 'item_prefix', 'item_prefix_joint', 'csa')
SUITES = dict(
    main=(*REFERENCES, 'full'),
    mechanism=('full', 'no_user_graph', 'no_prefix_graph', 'no_semantic_graphs',
               'no_user_matching', 'no_prefix_matching', 'no_matching'),
    structure=('full', 'prefix_only', 'user_only', 'joint_graph', 'mean_fusion', 'gate_fusion', 'random_user_graph'),
    views=('full', 'fixed_user_codes', 'no_sparse'),
    scoring=('full', 'score_user', 'score_item', 'score_continuous'),
    rvq=('full', 'offline_rvq'),
)
SUITES['core'] = tuple(dict.fromkeys(SUITES['main'] + SUITES['mechanism']))
SUITES['all'] = (*REFERENCES, *OVERRIDES)

# The completed prefix10-v2 suites above retain their original linear Full.
FOLLOWUP_REVISION = 'item-prefix-csa-mean-followup-v3'
FOLLOWUP_OVERRIDES = {
    'mean_full': (),
    'mean_no_user_graph': ('shadow.model.attribute.user_augmentation=false',),
    'mean_random_user_graph': ('shadow.model.attribute.random_user_graph=true',),
    'mean_no_user_matching': ('attack.model.attribute.user_matching=false',),
    'mean_user_idf': ('shadow.model.attribute.user_edge_weighting=idf',),
    'independent_social': (),
}
for percent, prefix in ((50, .05), (20, .02), (10, .01)):
    scale = (f'shadow.model.attribute.prefix_edge_scale={prefix}',
             f'shadow.model.attribute.user_edge_scale={percent / 100}')
    FOLLOWUP_OVERRIDES[f'mean_edges_p{percent}'] = scale
    FOLLOWUP_OVERRIDES[f'mean_idf_edges_p{percent}'] = (
        'shadow.model.attribute.user_edge_weighting=idf', *scale)
CONDITION_OVERRIDES = {**OVERRIDES, **FOLLOWUP_OVERRIDES}
SUITES.update(
    mean_user=('mean_full', 'mean_no_user_graph', 'mean_random_user_graph', 'mean_no_user_matching'),
    edge_weights=('mean_full', *(f'mean_edges_p{p}' for p in (50, 20, 10)),
                  'mean_user_idf', *(f'mean_idf_edges_p{p}' for p in (50, 20, 10))),
    social_split=('mean_full', 'independent_social'),
)
SUITES['followup'] = tuple(dict.fromkeys(SUITES['mean_user'] + SUITES['edge_weights'] + SUITES['social_split']))


def experiment_revision(conditions):
    return FOLLOWUP_REVISION if any(name in FOLLOWUP_OVERRIDES for name in conditions) else EXPERIMENT_REVISION


def contrasts(conditions):
    pairs = [('full', name) for name in conditions if name != 'full']
    pairs += [('no_prefix_graph', 'no_semantic_graphs'), ('no_user_graph', 'no_semantic_graphs'),
              ('no_prefix_matching', 'no_matching'), ('no_user_matching', 'no_matching')]
    pairs += [(name, 'mean_full') for name in SUITES['followup'] if name != 'mean_full']
    pairs += [('mean_user_idf', 'mean_full')]
    pairs += [(f'mean_idf_edges_p{p}', f'mean_edges_p{p}') for p in (50, 20, 10)]
    pairs += [(f'mean_idf_edges_p{p}', 'mean_user_idf') for p in (50, 20, 10)]
    return [pair for pair in dict.fromkeys(pairs) if all(name in conditions for name in pair)]
