from sralt.target.data.dataset import BaseTargetDataset, TrainTargetDataset, EvalTargetDataset


class BaseShadowDataset(BaseTargetDataset):
    """
    Base dataset class for shadow social recommendation models.
    """

    pass


class TrainShadowDataset(BaseShadowDataset, TrainTargetDataset):
    """
    Dataset class for training data in shadow social recommendation models.
    """

    pass


class EvalShadowDataset(BaseShadowDataset, EvalTargetDataset):
    """
    Dataset class for evaluation data in shadow social recommendation models.
    """

    pass
