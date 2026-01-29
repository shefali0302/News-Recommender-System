import torch
import torch.nn as nn

from models.ltc_encoder import LTCEncoder
from models.embeddings import JointEmbedding


class ShortTermEmbedding(nn.Module):
    """
    Short-term preference modeling (without LTC).

    Responsibilities:
    - Embedding lookup for recent interactions
    - Apply hybrid mask
    - Build masked interaction sequence X
    - Return (X, delta_t) for LTC
    """

    def __init__(
        self,
        num_news: int,
        num_categories: int,
        news_dim: int = 64,
        category_dim: int = 16
    ):
        super().__init__()

        self.embedding_layer = JointEmbedding(
            num_news=num_news,
            num_categories=num_categories,
            news_dim=news_dim,
            category_dim=category_dim
        )

        self.output_dim = news_dim + category_dim

    def forward(self, short_term_sequence):
        """
        Args:
            short_term_sequence:[(news_idx, category_idx, delta_t, mask),...]
        Returns:
            X: Tensor of shape (N, D)   -> masked interaction embeddings
            delta_t: Tensor of shape (N,) -> time gaps between interactions
        """

        news_ids = torch.tensor(
            [x[0] for x in short_term_sequence],
            dtype=torch.long
        ).unsqueeze(0)  # (1, N)

        category_ids = torch.tensor(
            [x[1] for x in short_term_sequence],
            dtype=torch.long
        ).unsqueeze(0)  # (1, N)

        delta_t = torch.tensor(
            [x[2] for x in short_term_sequence],
            dtype=torch.float32
        )  # (N,)

        mask = torch.tensor(
            [x[3] for x in short_term_sequence],
            dtype=torch.float32
        ).unsqueeze(-1)  # (N, 1)

        # Embedding lookup
        emb = self.embedding_layer(news_ids, category_ids)  # (1, N, D)
        emb = emb.squeeze(0)                                # (N, D)

        # Apply hybrid mask
        X = emb * mask                                     # (N, D)

        return X, delta_t


class ShortTermLTC(nn.Module):
    """
    Complete short-term preference pipeline:
    - Embedding extraction
    - LTC encoding
    
    Combines ShortTermModel and LTCEncoder in a single end-to-end module.
    """
    
    def __init__(
        self,
        num_news: int,
        num_categories: int,
        news_dim: int = 64,
        category_dim: int = 16,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.short_term_embedding = ShortTermEmbedding(
            num_news=num_news,
            num_categories=num_categories,
            news_dim=news_dim,
            category_dim=category_dim
        )
        
        embedding_dim = news_dim + category_dim
        self.ltc_encoder = LTCEncoder(embedding_dim, hidden_dim)
    
    def forward(self, short_term_sequence):
        """
        Args:
            short_term_sequence:[(news_idx, category_idx, delta_t, mask), ...]
        
        Returns:
            encoded: LTC encoded user representation of shape (hidden_dim,)
            X: Tensor of shape (N, D) - masked interaction embeddings (for inspection)
            delta_t: Tensor of shape (N,) - time gaps (for inspection)
        """
        X, delta_t = self.short_term_embedding(short_term_sequence)
        encoded = self.ltc_encoder(X, delta_t)
        
        return encoded, X, delta_t
    

