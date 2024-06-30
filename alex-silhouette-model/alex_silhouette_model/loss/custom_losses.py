import torch
import torch.nn as nn
import numpy as np


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