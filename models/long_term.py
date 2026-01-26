import torch
import torch.nn as nn

from models.embeddings import JointEmbedding


class LongTermModel(nn.Module):
    """
    Long-term preference modeling (without LTC).

    Responsibilities:
    - Embedding lookup for interactions in each day
    - Mean pooling within daily chunks
    - Build daily preference sequence Z
    - Return (Z, delta_t_days) for LTC
    """

    def __init__(
        self,
        num_news: int,
        num_categories: int,
        news_dim: int = 64,
        category_dim: int = 16
    ):
        super().__init__()

        # Shared embedding layer (news + category)
        self.embedding_layer = JointEmbedding(
            num_news=num_news,
            num_categories=num_categories,
            news_dim=news_dim,
            category_dim=category_dim
        )

        # Useful later for LTC / fusion
        self.output_dim = news_dim + category_dim

    def forward(self, long_term_sequence):
        """
        Args:
            long_term_sequence:[(daily_interactions, delta_t_days),...]
            daily_interactions: [(news_idx, category_idx), ...]
        Returns:
            Z: Tensor of shape (M, D)   -> daily preference vectors
            delta_t: Tensor of shape (M,) -> time gap between days
        """

        daily_vectors = []
        day_gaps = []

        for daily_interactions, delta_days in long_term_sequence:

            # -------------------------------
            # Build ID tensors for one day
            # -------------------------------
            news_ids = torch.tensor(
                [x[0] for x in daily_interactions],
                dtype=torch.long
            ).unsqueeze(0)  # (1, N_m)

            category_ids = torch.tensor(
                [x[1] for x in daily_interactions],
                dtype=torch.long
            ).unsqueeze(0)  # (1, N_m)

            # -------------------------------
            # Embedding lookup
            # -------------------------------
            interaction_emb = self.embedding_layer(
                news_ids, category_ids
            )  # (1, N_m, D)

            interaction_emb = interaction_emb.squeeze(0)        # (N_m, D)

            # -------------------------------
            # Mean pooling within the day
            # -------------------------------
            daily_vector = interaction_emb.mean(dim=0)          # (D,)
            daily_vectors.append(daily_vector)

            day_gaps.append(delta_days)

        # -------------------------------
        # Build daily sequence
        # -------------------------------
        Z = torch.stack(daily_vectors, dim=0)                   # (M, D)
        delta_t = torch.tensor(day_gaps, dtype=torch.float32)   # (M,)

        return Z, delta_t
