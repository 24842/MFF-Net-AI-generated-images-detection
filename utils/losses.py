import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.a, self.g, self.r = alpha, gamma, reduction

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        pt = torch.exp(-bce)
        loss = self.a * (1 - pt) ** self.g * bce
        if self.r == 'mean': return loss.mean()
        elif self.r == 'sum': return loss.sum()
        return loss