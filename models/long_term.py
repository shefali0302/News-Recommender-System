# -------------------------------------------------------------------------------------------
# This file defines the long-term preference modeling components, including:
# - LongTermEmbedding: extracts daily preference vectors from long-term interaction history
# - LongTermLTC: combines the embedding extraction with the LTC encoder for temporal modeling
# -------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from models.ltc_encoder import LTCEncoder


class LongTermEmbedding(nn.Module):
    """
    Long-term preference modeling (without LTC).

    Responsibilities:
    - Embedding lookup for interactions in each day
    - Time-aware weighted pooling within each day
    - Build daily preference sequence Z
    - Return (Z, delta_t_days) for LTC
    """
    def __init__(self, joint_embedding):
        super().__init__()
        self.embedding_layer = joint_embedding
        self.output_dim = joint_embedding.output_dim
        self.debug_done = False
        self.attn = nn.Linear(self.output_dim, 1)


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
            if len(daily_interactions) == 0:
                continue
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
            # Time aware pooling within the day
            # -------------------------------
            delta_t = torch.tensor(
                [x[2] for x in daily_interactions],
                dtype=torch.float32,
                device=device
            )

            delta_t = delta_t / 3600.0
            delta_t = torch.log1p(delta_t)

            weights = torch.exp(-delta_t).unsqueeze(-1)
            weights = weights / (weights.sum() + 1e-8)

            daily_vector = torch.sum(interaction_emb * weights, dim = 0) # (D,)

            daily_vectors.append(daily_vector)

            day_gaps.append(delta_days)

            if not self.debug_done:
                print("\n===== LONG TERM DEBUG =====")
                print("interaction delta_t:", delta_t[:5])
                print("weights:", weights[:5])
                print("interaction_emb shape:", interaction_emb.shape)
                self.debug_done = True
        
        # -------------------------------
        # Build daily sequence
        # -------------------------------
        if len(daily_vectors) == 0:
            return None, None
        Z = torch.stack(daily_vectors, dim=0)                   # (M, D)
        attn_scores = self.attn(Z).squeeze(-1)  # (M,)
        attn_weights = torch.softmax(attn_scores, dim=0).unsqueeze(-1)  # (M,1)
        Z = Z * attn_weights # (M, D)
        Z = Z / (attn_weights.sum() + 1e-8)

        delta_days_tensor  = torch.tensor(day_gaps, dtype=torch.float32, device=device)   # (M,)

        delta_days_tensor = torch.log1p(delta_days_tensor)

        if not self.debug_done:
            print("day attn_weights:", attn_weights[:5])


        return Z, delta_days_tensor 

class LongTermLTC(nn.Module):
    """
    Complete long-term preference pipeline:
    - Embedding extraction
    - LTC encoding
    
    Combines LongTermEmbedding and LTCEncoder in a single end-to-end module.
    """
    
    def __init__(self, joint_embedding, hidden_dim: int = 64):
        super().__init__()

        self.long_term_embedding = LongTermEmbedding(joint_embedding)

        embedding_dim = joint_embedding.output_dim
        self.ltc_encoder = LTCEncoder(embedding_dim, hidden_dim)

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.pooling = nn.Linear(hidden_dim, 1)


    def forward(self, long_term_sequence):
        Z, delta_days_tensor = self.long_term_embedding(long_term_sequence)
        if Z is None:
            return None, None, None

        encoded_seq = self.ltc_encoder(Z, delta_days_tensor)

        attn_output, _ = self.self_attn(
            encoded_seq,
            encoded_seq,
            encoded_seq
        )

        pool_scores = self.pooling(attn_output).squeeze(-1)
        pool_weights = torch.softmax(pool_scores, dim=1).unsqueeze(-1)

        user_vector = torch.sum(
            attn_output * pool_weights,
            dim=1
        )
        
        return user_vector, Z, delta_days_tensor


