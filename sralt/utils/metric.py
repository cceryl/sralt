import torch
from torchmetrics import Metric


class TPRatFPR(Metric):
    """
    True Positive Rate (TPR) at a specified False Positive Rate (FPR).
    """
    full_state_update = True

    def __init__(self, target_fpr: float):
        """
        Initialize the TPRatFPR metric.

        :param target_fpr: The target false positive rate (between 0 and 1).
        """
        super().__init__(dist_sync_on_step=False)
        assert 0 <= target_fpr <= 1

        self.target_fpr = target_fpr
        self.preds: list[torch.Tensor]
        self.target: list[torch.Tensor]
        self.add_state('preds', default=[], dist_reduce_fx='cat')
        self.add_state('target', default=[], dist_reduce_fx='cat')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric state with new predictions and targets.

        :param preds: Predicted probabilities of shape (batch_size,).
        :param target: Ground truth binary labels of shape (batch_size,).
        """
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        """
        Compute the TPR at the specified FPR.

        :return: The TPR at the specified FPR.
        """
        # Sort predictions and targets by predicted scores in descending order
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)

        sorted_idx = torch.argsort(preds, descending=True)
        preds = preds[sorted_idx]
        target = target[sorted_idx]

        # Calculate TPR and FPR
        pos = torch.sum(target == 1).item()
        neg = torch.sum(target == 0).item()

        if pos == 0 or neg == 0:
            return torch.tensor(float('nan'), device=preds.device)

        tp = torch.cumsum(target == 1, dim=0)
        fp = torch.cumsum(target == 0, dim=0)

        tpr = tp / pos
        fpr = fp / neg

        # Corner cases
        if fpr[-1] < self.target_fpr:
            return tpr[-1]

        if fpr[0] >= self.target_fpr:
            return tpr[0]

        # Linear interpolation to find TPR at target FPR
        idx = torch.searchsorted(fpr, torch.tensor(self.target_fpr, device=preds.device), right=False)

        fpr0, fpr1 = fpr[idx-1], fpr[idx]
        tpr0, tpr1 = tpr[idx-1], tpr[idx]

        if fpr1 == fpr0:
            return tpr1

        alpha = (self.target_fpr - fpr0) / (fpr1 - fpr0)
        result = tpr0 + alpha * (tpr1 - tpr0)
        return result
