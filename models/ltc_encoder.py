# ----------------------------------------------------------------------------------------------
# This file defines the LTCEncoder, which is a wrapper around the LTC module from ncps.torch.
# It takes in the embedded interaction sequences and their corresponding time gaps, and produces 
# a fixed-size user representation that captures temporal dynamics.
# -----------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import FullyConnected


class LTCEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        ode_unfolds=6,
        mixed_memory=False,
        epsilon=1e-8
    ):
        super().__init__()

        self.input_dim = input_dim + 1  # +1 for delta_t
        self.hidden_dim = hidden_dim

        wiring = FullyConnected(hidden_dim)

        self.ltc = LTC(
            input_size=self.input_dim,
            units=wiring,
            return_sequences=False,
            batch_first=True,
            mixed_memory=mixed_memory,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon
        )

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, X, delta_t):
        """
        X: (seq_len, dim)
        delta_t: (seq_len,)
        """

        if X.dim() == 2:
            X = X.unsqueeze(0)

        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(0)

        # normalize delta_t for stability
        delta_t = torch.log1p(delta_t)

        # append time gap feature
        delta_t = delta_t.unsqueeze(-1)  # (batch, seq, 1)

        X_aug = torch.cat([X, delta_t], dim=-1)

        batch_size = X_aug.size(0)

        h0 = torch.zeros(batch_size, self.hidden_dim).to(X.device)

        _, h_final = self.ltc(X_aug, h0)

        return h_final.squeeze(0)

    # -------------------------
    # Parameter Debug Tools
    # -------------------------
    def print_ltc_parameters(self):
        print("\n==== LTC Parameters ====")
        for name, p in self.ltc.named_parameters():
            print(name, p.shape)

    def get_ltc_parameters(self):
        return dict(self.ltc.named_parameters())


# import torch
# import torch.nn as nn
# from ncps.torch import LTC


# class LTCEncoder(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim):
#         super().__init__()

#         self.hidden_dim = hidden_dim

#         self.ltc = LTC(
#             input_size=embedding_dim,
#             units=hidden_dim,
#             batch_first=True
#         )

#     def forward(self, x, delta_t):
#         """
#         Args:
#             x: Tensor (N, D) or (1, N, D)
#             delta_t: Tensor (N,)
#         Returns:
#             encoded: Tensor (hidden_dim,)
#         """

#         # Ensure batch dimension
#         if x.dim() == 2:
#             x = x.unsqueeze(0)          # (1, N, D)

#         # LTC expects timespans between steps
#         # Clamp to avoid numerical instability
#         timespans = delta_t.clamp(min=1e-3).unsqueeze(0)  # (1, N)

#         outputs, final_hidden = self.ltc(
#             x,
#             timespans=timespans
#         )

#         # final_hidden shape: (1, hidden_dim)
#         return final_hidden.squeeze(0)
