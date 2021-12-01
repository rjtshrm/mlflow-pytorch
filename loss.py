import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, l2_weight=1, l1_weight=1) -> None:
        super().__init__()
        self.l2_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")

        self.l2_weight = l2_weight
        self.l1_weight = l1_weight

    def forward(self, output, target):
        return self.l2_weight * self.l2_loss(output, target) + self.l1_weight * self.l1_loss(output, target)