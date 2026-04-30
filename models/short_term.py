# --------------------------------------------------------------------------------------------
# This file defines the short-term preference modeling components, including:
# - ShortTermEmbedding: extracts masked interaction embeddings from recent interactions
# - ShortTermLTC: combines the embedding extraction with the LTC encoder for temporal modeling
# --------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from models.ltc_encoder import LTCEncoder


class ShortTermEmbedding(nn.Module):
    """
    Short-term preference modeling (without LTC).

    Responsibilities:
    - Embedding lookup for recent interactions
    - Apply time-aware weighting
    - Build weighted interaction sequence X
    - Return (X, delta_t) for LTC
    """

    def __init__(self, joint_embedding):
        super().__init__()
        self.embedding_layer = joint_embedding
        self.output_dim = joint_embedding.output_dim
        self.debug_done = False

    
    def forward(self, short_term_sequence):
        """
        Args:
            short_term_sequence:[(news_idx, timestamp, category_idx, delta_t),...]
        Returns:
            X: Tensor of shape (N, D)   -> time-weighted interaction embeddings
            delta_t: Tensor of shape (N,) -> time gaps between interactions
        """
        device = next(self.parameters()).device

        news_ids = torch.tensor(
            [x[0] for x in short_term_sequence],
            dtype=torch.long,
            device=device
        ).unsqueeze(0) # (1, N)

        category_ids = torch.tensor(
            [x[2] for x in short_term_sequence],
            dtype=torch.long,
            device=device
        ).unsqueeze(0)  # (1, N)

        N = len(short_term_sequence)

        # Embedding lookup
        emb = self.embedding_layer(news_ids, category_ids)  # (1, N, D)
        emb = emb.squeeze(0)                                # (N, D)
        
        delta_t = torch.tensor(
            [x[3] for x in short_term_sequence],
            dtype=torch.float32,
            device=device
        )
        weights = torch.exp(-delta_t.clamp(max=50)).unsqueeze(-1)
        weights = weights / (weights.sum() + 1e-8)

        X = emb * weights 


        if not self.debug_done:
            print("\n===== SHORT TERM DEBUG =====")
            print("delta_t:", delta_t[:5])
            print("weights:", weights[:5])
            print("X shape:", X.shape)
            self.debug_done = True

        return X, delta_t


class ShortTermLTC(nn.Module):
    """
    Complete short-term preference pipeline:
    - Embedding extraction
    - LTC encoding
    
    Combines ShortTermModel and LTCEncoder in a single end-to-end module.
    """
    
    def __init__(self, joint_embedding, hidden_dim: int = 64):
        super().__init__()

        self.short_term_embedding = ShortTermEmbedding(joint_embedding)

        embedding_dim = joint_embedding.output_dim
        self.ltc_encoder = LTCEncoder(embedding_dim, hidden_dim)

    
    def forward(self, short_term_sequence):
        """
        Args:
            short_term_sequence:[(news_idx, timestamp category_idx, delta_t), ...]
        
        Returns:
            encoded: LTC encoded user representation of shape (hidden_dim,)
            X: Tensor of shape (N, D) - masked interaction embeddings (for inspection)
            delta_t: Tensor of shape (N,) - time gaps (for inspection)
        """
        X, delta_t = self.short_term_embedding(short_term_sequence)
        encoded = self.ltc_encoder(X, delta_t)
        
        return encoded, X, delta_t
    

