# ----------------------------------------------------------------------------------------------
# This file defines the LTCEncoder, which is a wrapper around the LTC module from ncps.torch.
# It takes in the embedded interaction sequences and their corresponding time gaps, and produces 
# a fixed-size user representation that captures temporal dynamics.
# -----------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from ncps.torch import LTC


class LTCEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.ltc = LTC(
            input_size=embedding_dim,
            units=hidden_dim,
            batch_first=True
        )

    def forward(self, x, delta_t):
        """
        Args:
            x: Tensor (N, D) or (1, N, D)
            delta_t: Tensor (N,)
        Returns:
            encoded: Tensor (hidden_dim,)
        """

        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)          # (1, N, D)

        # LTC expects timespans between steps
        # Clamp to avoid numerical instability
        timespans = delta_t.clamp(min=1e-3).unsqueeze(0)  # (1, N)

        outputs, final_hidden = self.ltc(
            x,
            timespans=timespans
        )

        # final_hidden shape: (1, hidden_dim)
        return final_hidden.squeeze(0)
