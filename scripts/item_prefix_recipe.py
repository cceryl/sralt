"""Stage-B condition definitions; standard-library only for scheduler/reporting."""

EXPERIMENT_REVISION = 'item-prefix-edge10-library-rvq-v3'
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


# Current Full uses .1 prefix edge scale; historical results keep their saved configs.
RATIOS = (50, 30, 10, 5, 1)
SOCIAL_EPOCHS = (1, 5, 10, 20, 50)
OVERRIDES['rvq_end_to_end'] = ()
for epochs in SOCIAL_EPOCHS:
    OVERRIDES[f'social_e{epochs}'] = (f'+shadow.model.attribute.social_fixed_epochs={epochs}',)
for percent in RATIOS:
    OVERRIDES[f'edge_p{percent}'] = (f'shadow.model.attribute.prefix_edge_scale={percent / 100}',)
    OVERRIDES[f'mix_p{percent}'] = ('+shadow.model.attribute.graph_normalization=independent',
                                  f'+shadow.model.attribute.prefix_mix={percent / 100}')
SUITES.update(
    rvq=('item_prefix', 'rvq_end_to_end'),
    social_epochs=('item_prefix', *(f'social_e{e}' for e in SOCIAL_EPOCHS)),
    edge_scale=('item_prefix', *(f'edge_p{p}' for p in RATIOS)),
    independent_mix=('item_prefix', *(f'mix_p{p}' for p in RATIOS)),
)
SUITES['followup'] = ('item_prefix', 'rvq_end_to_end',
                      *(f'social_e{e}' for e in SOCIAL_EPOCHS),
                      *(f'edge_p{p}' for p in RATIOS), *(f'mix_p{p}' for p in RATIOS))


COMMITMENT_WEIGHTS = (.01, .1, .25, 1.)
RVQ_CONDITIONS = {}
for ema in (False, True):
    for weight in COMMITMENT_WEIGHTS:
        if weight == .25:
            name = 'rvq_ema' if ema else 'rvq_end_to_end'
        else:
            name = f"rvq_{'ema' if ema else 'grad'}_cw{str(weight).replace('.', 'p')}"
        RVQ_CONDITIONS[(ema, weight)] = name
        OVERRIDES[name] = (f'shadow.model.attribute.rvq_ema_update={str(ema).lower()}',
                           f'shadow.model.attribute.rvq_commitment_weight={weight}')
SUITES.update(
    rvq_ema=('item_prefix', 'rvq_end_to_end', 'rvq_ema'),
    rvq_commitment_grad=('item_prefix', *(RVQ_CONDITIONS[(False, w)] for w in COMMITMENT_WEIGHTS)),
    rvq_commitment_ema=('item_prefix', *(RVQ_CONDITIONS[(True, w)] for w in COMMITMENT_WEIGHTS)),
    rvq_hparams=('item_prefix', *RVQ_CONDITIONS.values()),
)


def contrasts(conditions):
    pairs = [('item_prefix', 'baseline'), ('item_prefix', 'tail_proxy'), ('tail_proxy', 'baseline'),
             ('item_prefix', 'no_graph'), ('item_prefix', 'no_matching'),
             ('item_prefix', 'identity'), ('item_prefix', 'frequency_off'),
             ('full_no_sparse', 'no_graph_no_sparse')]
    pairs.extend((name, 'item_prefix') for name in SUITES['followup'][1:])
    pairs.extend((name, 'item_prefix') for name in RVQ_CONDITIONS.values())
    pairs.extend((RVQ_CONDITIONS[(True, w)], RVQ_CONDITIONS[(False, w)]) for w in COMMITMENT_WEIGHTS)
    for ema in (False, True):
        pairs.extend((RVQ_CONDITIONS[(ema, w)], RVQ_CONDITIONS[(ema, .25)])
                     for w in COMMITMENT_WEIGHTS if w != .25)
    return list(dict.fromkeys(pair for pair in pairs if all(name in conditions for name in pair)))


def experiment_revision(conditions):
    # Edge10 default and RVQ hyperparameters define a new batch identity.
    # Never resume historical edge100 or mixed v1/v2 runs under these defaults.
    return EXPERIMENT_REVISION
