"""CSA + ItemPrefix complete-expert ensemble and frozen representation controls."""

from .models import (FrozenScoreEnsemble, DegreeGatedEnsemble,
                     AttackRepresentationFusion, ShadowUserFusion)
from .evidence import ExpertBundle, EnsembleEvidence
from .training import load_oof_attack_evidence, load_fusion_model, load_fusion_pair_pool

__all__ = ['ExpertBundle', 'EnsembleEvidence', 'FrozenScoreEnsemble',
           'DegreeGatedEnsemble', 'AttackRepresentationFusion', 'ShadowUserFusion',
           'load_oof_attack_evidence', 'load_fusion_model', 'load_fusion_pair_pool']
