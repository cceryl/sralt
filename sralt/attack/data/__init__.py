from .data import AttackData
from .dataset import TrainAttackDataset, EvalAttackDataset
from sralt.target.data import TargetData
from sralt.shadow.data import ShadowData
from sralt.shadow.model import ShadowModel
from sralt.utils import disk_cache, set_seed 


@disk_cache(
    cache_dir='.cache/attack/data/{target_data.name}-{shadow_data.target_model_name}-{shadow_model.__class__.__name__}',
    hit='Loading cached attack data {target_data.name} from "{__cache__.path}".',
    miss='Generating attack data with data "{target_data.name}" and caching to "{__cache__.path}".'
)
def load_attack_data(
    seed: int, deterministic: bool,
    target_data: TargetData,
    shadow_data: ShadowData,
    shadow_model: ShadowModel,
    val_ratio: float
) -> AttackData:
    """
    Load or generate attack data.

    :param seed: The random seed for attack data generation.
    :param deterministic: Whether to use deterministic behavior for attack data generation.
    :param target_data: The target data.
    :param shadow_data: The shadow data.
    :param shadow_model: The shadow model.
    :param val_ratio: The ratio of validation edges in the shadow social graph.
    :return: An instance of the attack data.
    """
    set_seed(seed, deterministic=deterministic)

    return AttackData(
        target_data=target_data,
        shadow_data=shadow_data,
        shadow_model=shadow_model,
        val_ratio=val_ratio
    )


__all__ = [
    'load_attack_data',
    'AttackData', 'TrainAttackDataset', 'EvalAttackDataset'
]
