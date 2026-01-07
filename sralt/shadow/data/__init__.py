from .data import ShadowData
from .dataset import TrainShadowDataset, EvalShadowDataset
from sralt.target.data import TargetData
from sralt.target.model import TargetModel
from sralt.utils import disk_cache, set_seed


@disk_cache(
    cache_dir='.cache/shadow/data/{target_data.name}-{target_model.__class__.__name__}',
    hit_log_format='Loading cached shadow data {target_data.name} from "{__cache__.path}".',
    miss_log_format='Generating shadow data from target data "{target_data.name}" and caching to "{__cache__.path}".'
)
def load_shadow_data(
    seed: int, deterministic: bool,
    target_data: TargetData, target_model: TargetModel,
    shadow_social_ratio: float, num_recommendations: int,
    inference_batch_size: int = 1024
) -> ShadowData:
    """
    Load or generate shadow data.

    :param seed: The random seed for shadow data generation.
    :param deterministic: Whether to use deterministic behavior for shadow data generation.
    :param target_data: The target data.
    :param target_model: The target recommendation model.
    :param shadow_social_ratio: The ratio of social edges to include in the shadow data.
    :param num_recommendations: The number of recommendations to generate for each user.
    :param inference_batch_size: The batch size for generating recommendations.
    :return: An instance of the shadow data.
    """
    set_seed(seed, deterministic=deterministic)

    return ShadowData(
        target_data=target_data,
        target_model=target_model,
        shadow_social_ratio=shadow_social_ratio,
        num_recommendations=num_recommendations,
        inference_batch_size=inference_batch_size
    )


__all__ = [
    'load_shadow_data',
    'ShadowData',  'TrainShadowDataset', 'EvalShadowDataset'
]
