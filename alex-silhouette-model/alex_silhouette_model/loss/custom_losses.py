import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class HammingDistance(nn.Module):
    def __init__(self):
        super(HammingDistance, self).__init__()


    def forward(self, input, target):
        # TODO set the threshold per (trianable) parameter?
        # Convert to binary (0 or 1)
        input_binary = torch.where(input >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
        target_binary = torch.where(target >= 0.5, torch.tensor(1.0), torch.tensor(0.0))

        # Calculate Hamming distance
        hamming_distance = torch.sum(input_binary != target_binary).item()
        # Alternativley:  torch.abs(input_binary - target_binary).mean().item()

        return hamming_distance


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss


# Copied from
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# Copied from
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
ALPHA = 0.5
BETA = 0.5


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky