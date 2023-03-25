import torch.nn as nn
import torch

class mpjpe_loss(nn.Module):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    returns mean error across all data points
    and mean per joint error 17 x 1
    """
    def __init__(self):
        super(mpjpe_loss, self).__init__()
    
    def forward(self, predicted, target):
        assert predicted.shape == target.shape
        err = torch.norm(predicted - target, dim=len(target.shape)-1) # num_batch x num_joint
        return torch.mean(err)


