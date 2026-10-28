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


def contrasts(conditions):
    pairs = [('full', name) for name in conditions if name != 'full']
    pairs += [('no_prefix_graph', 'no_semantic_graphs'), ('no_user_graph', 'no_semantic_graphs'),
              ('no_prefix_matching', 'no_matching'), ('no_user_matching', 'no_matching')]
    return [pair for pair in pairs if all(name in conditions for name in pair)]
