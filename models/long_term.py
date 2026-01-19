import torch
import torch.nn as nn

from models.embeddings import NewsCategoryEmbedding


class LongTermModel(nn.Module):
    """
    Long-term preference modeling (without LTC).

    Tasks implemented:
    - Embedding lookup 
    - Mean pooling within daily chunks 
    - Build daily vector sequence Z 
    """

    def __init__(self, num_news, num_categories, emb_dim):
        super().__init__()

        self.embedding_layer = NewsCategoryEmbedding(
            num_news=num_news,
            num_categories=num_categories,
            emb_dim=emb_dim
        )

    def forward(self, long_term_sequence):
        """
        Args:
            long_term_sequence: [(daily_interactions, delta_t_days),...]
            daily_interactions: [(news_id, category, delta_t), ...]
        """

        daily_vectors = []
        day_gaps = []

        for daily_interactions, delta_days in long_term_sequence:
            # ---- Embedding lookup ----
            news_ids = torch.tensor(
                [x[0] for x in daily_interactions],
                dtype=torch.long
            )
            category_ids = torch.tensor(
                [x[1] for x in daily_interactions],
                dtype=torch.long
            )

            interaction_emb = self.embedding_layer(
                news_ids, category_ids
            )  # shape: [N_m, D]

            # ---- Mean pooling ----
            daily_vector = interaction_emb.mean(dim=0)
            daily_vectors.append(daily_vector)

            day_gaps.append(delta_days)

        # ---- Build daily vector sequence ----
        Z = torch.stack(daily_vectors, dim=0)          # [M, D]
        delta_t = torch.tensor(day_gaps, dtype=torch.float32)  # [M]

        # For now, return intermediate outputs
        return Z, delta_t
