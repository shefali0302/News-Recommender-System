# -------------------------------------------------------------------------------------------
# This file defines the long-term preference modeling components, including:
# - LongTermEmbedding: extracts daily preference vectors from long-term interaction history
# - LongTermLTC: combines the embedding extraction with the LTC encoder for temporal modeling
# -------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

from models.embeddings import JointEmbedding
from models.ltc_encoder import LTCEncoder


class LongTermEmbedding(nn.Module):
    """
    Long-term preference modeling (without LTC).

    Responsibilities:
    - Embedding lookup for interactions in each day
    - Mean pooling within daily chunks
    - Build daily preference sequence Z
    - Return (Z, delta_t_days) for LTC
    """
    def __init__(self, joint_embedding):
        super().__init__()
        self.embedding_layer = joint_embedding
        self.output_dim = joint_embedding.news_dim + joint_embedding.category_dim



    def forward(self, long_term_sequence):
        """
        Args:
            long_term_sequence:[(daily_interactions, delta_t_days),...]
            daily_interactions: [(news_idx, category_idx, delta_t), ...]
        Returns:
            Z: Tensor of shape (M, D)   -> daily preference vectors
            delta_t: Tensor of shape (M,) -> time gap between days
        """
        device = next(self.parameters()).device

        daily_vectors = []
        day_gaps = []

        for daily_interactions, delta_days in long_term_sequence:

            # -------------------------------
            # Build ID tensors for one day
            # -------------------------------
            news_ids = torch.tensor(
                [x[0] for x in daily_interactions],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)  # (1, N_m)

            category_ids = torch.tensor(
                [x[1] for x in daily_interactions],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)  # (1, N_m)

            # -------------------------------
            # Embedding lookup
            # -------------------------------
            interaction_emb = self.embedding_layer(
                news_ids, category_ids
            )  # (1, N_m, D)
            interaction_emb = interaction_emb.squeeze(0) # (N_m, D)

            # -------------------------------
            # Recency-weighted pooling 
            # -------------------------------
            delta_t_list = torch.tensor(
                [x[2] for x in daily_interactions],
                dtype=torch.float32,
                device=device
            )  # (N_m,)
            delta_t_list = torch.log1p(delta_t_list)  # log-transform to reduce skewness
            weights = torch.exp(-delta_t_list)  # (N_m,)
            weights = weights / (weights.sum() + 1e-8)  # (N_m,)
            weights = weights.unsqueeze(-1)  # (N_m, 1)
            daily_vector = torch.sum(interaction_emb*weights, dim = 0) # (D,)

            daily_vectors.append(daily_vector)

            day_gaps.append(delta_days)

        # -------------------------------
        # Build daily sequence
        # -------------------------------
        Z = torch.stack(daily_vectors, dim=0)                   # (M, D)
        delta_t = torch.tensor(day_gaps, dtype=torch.float32, device=device)   # (M,)

        return Z, delta_t

class LongTermLTC(nn.Module):
    """
    Complete short-term preference pipeline:
    - Embedding extraction
    - LTC encoding
    
    Combines ShortTermModel and LTCEncoder in a single end-to-end module.
    """
    
    def __init__(self, joint_embedding, hidden_dim: int = 64):
        super().__init__()

        self.long_term_embedding = LongTermEmbedding(joint_embedding)

        embedding_dim = joint_embedding.news_dim + joint_embedding.category_dim
        self.ltc_encoder = LTCEncoder(embedding_dim, hidden_dim)


    def forward(self, long_term_sequence):
        Z, delta_t = self.long_term_embedding(long_term_sequence)
        encoded = self.ltc_encoder(Z, delta_t)
        
        return encoded, Z, delta_t


