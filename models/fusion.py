# --------------------------------------------------------------------------------------
# This file defines the FusionGate module, which is responsible for dynamically fusing
# the short-term and long-term user representations. It uses a learnable gating mechanism
# to determine how much weight to give to each representation based on the current input.
# --------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class FusionGate(nn.Module):
    """
    Dynamic sequence-aware gating module
    Fuses short-term and long-term user representations
    """

    def __init__(self, dim=64):
        super(FusionGate, self).__init__()

        # Learnable parameters
        self.ws = nn.Parameter(torch.randn(dim) * 0.01)
        self.wl = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, st, lt):
        """
        st: Short-term vector (dim,)
        lt: Long-term vector (dim,)
        returns: fused user vector (dim,)
        """

        # Gate score (scalar)
        gate_score = torch.dot(self.ws, st) + torch.dot(self.wl, lt) + self.b

        # Sigmoid to keep gate between 0 and 1
        gt = torch.sigmoid(gate_score)

        # Fuse representations
        fused_user_vector = gt * st + (1 - gt) * lt

        return fused_user_vector, gt
