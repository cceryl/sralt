"""Stage-B condition definitions; standard-library only for scheduler/reporting."""

EXPERIMENT_REVISION = 'item-prefix-stage-b-v1'
MAIN = ('baseline', 'tail_proxy', 'item_prefix')
MECHANISM = ('item_prefix', 'no_graph', 'no_matching', 'identity', 'frequency_off',
             'full_no_sparse', 'no_graph_no_sparse')
SUITES = dict(main=MAIN, mechanism=MECHANISM, all=MAIN + MECHANISM[1:])

OVERRIDES = {
    'item_prefix': (),
    'no_graph': ('shadow.model.attribute.semantic_augmentation=false',),
    'no_matching': ('attack.model.attribute.prefix_matching=false',),
    'identity': ('shadow.model.attribute.proxy_kind=identity',),
    'frequency_off': ('shadow.model.attribute.frequency_weighting=false',),
    'full_no_sparse': ('attack.model.attribute.sparse_training=false',),
    'no_graph_no_sparse': ('shadow.model.attribute.semantic_augmentation=false',
                          'attack.model.attribute.sparse_training=false'),
}


def contrasts(conditions):
    pairs = [('item_prefix', 'baseline'), ('item_prefix', 'tail_proxy'), ('tail_proxy', 'baseline'),
             ('item_prefix', 'no_graph'), ('item_prefix', 'no_matching'),
             ('item_prefix', 'identity'), ('item_prefix', 'frequency_off'),
             ('full_no_sparse', 'no_graph_no_sparse')]
    return [pair for pair in pairs if all(name in conditions for name in pair)]
