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
        self.Ws = nn.Linear(dim, dim)
        self.Wl = nn.Linear(dim, dim)
        self.Wi = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(dim))


    def forward(self, st, lt):
        """
        st: (B, D)
        lt: (B, D)
        Returns:
            fused_user_vector: (B, D)
            gt: (B, D) - gate values for interpretability
        """

        interaction = st * lt  # (B, D)

        gt = torch.sigmoid(
            (self.Ws(st) +
            self.Wl(lt) +
            self.Wi(interaction) +
            self.bias) 
            / (st.shape[-1] ** 0.5)
        ) # (B, D)


        fused_user_vector = gt * st + (1 - gt) * lt  # (B, D)

        return fused_user_vector, gt

