import torch
import torch.nn as nn

from models.embeddings import JointEmbedding


class ShortTermModel(nn.Module):
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
            short_term_sequence:
                [
                  (news_idx, category_idx, delta_t, mask),
                  ...
                ]

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
