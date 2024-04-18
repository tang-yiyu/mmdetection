import torch
import torch.nn as nn
from torch import Tensor

from typing import List, Dict

from mmdet.registry import MODELS
# from mmdet.models.losses.utils import weighted_loss

def policy_loss(decisions_set: List, losses: Dict) -> Tensor:
    """Policy loss.

     Args:
        decisions_set (List): The decisions list.
        losses (Dict): The loss dict.

    Returns:
        Tensor: Calculated policy loss
    """
    policy_losses = []
    cost_weights = [1.0, 1.0] # The weight of each stream
    gammas = 10 # Penalize incorrect predictions
    incorrectness = losses['loss_cls'] + losses['loss_bbox']
    correctness = torch.ones_like(incorrectness) - incorrectness
    for decisions in decisions_set:
        policy_loss = torch.tensor(0.0, dtype=decisions.dtype, device=decisions.device)
        for w, pl in zip(cost_weights, decisions.chunk(chunks=2, dim=0)):
            loss = w * torch.mean(correctness * pl)
            policy_loss = policy_loss + loss
        policy_loss = policy_loss + torch.mean(incorrectness * gammas)
        policy_losses.append(policy_loss)
    loss_policy=sum(policy_losses) / len(policy_losses)
    return loss_policy

@MODELS.register_module()
class PolicyLoss(nn.Module):
    """Smooth L1 loss.

    Args:
        decisions_set (List): The decisions list.
        losses (Dict): The loss dict.
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 loss_weight: float = 0.05) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self,
                decisions_set: List,
                losses: Dict) -> Tensor:
        """Forward function.

        Args:
            decisions_set (List): The decisions list.
            losses (Dict): The loss dict.

        Returns:
            Tensor: Calculated policy loss
        """

        loss_policy = self.loss_weight * policy_loss(
            decisions_set,
            losses)
        return loss_policy