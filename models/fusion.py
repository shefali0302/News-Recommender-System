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
            st: (B, D)
            lt: (B, D)
            """

            # Compute gate score per user
            gate_score = (
                torch.matmul(st, self.ws) +
                torch.matmul(lt, self.wl) +
                self.b
            )  # (B,)

            gt = torch.sigmoid(gate_score).unsqueeze(-1)  # (B, 1)

            fused_user_vector = gt * st + (1 - gt) * lt  # (B, D)

            return fused_user_vector, gt

