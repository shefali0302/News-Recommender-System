"""
Trainable embedding layers
Responsible ONLY for mapping IDs -> dense vectors

Shared across:
- short-term modeling
- long-term modeling
"""

import torch
import torch.nn as nn


class NewsEmbedding(nn.Module):
    """
    Trainable embedding layer for news articles
    """

    def __init__(self, num_news: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_news,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

    def forward(self, news_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:
            news_ids: Tensor of shape (batch_size, seq_len)
        Output:
            news_embeddings: (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(news_ids)


class CategoryEmbedding(nn.Module):
    """
    Trainable embedding layer for news categories
    """

    def __init__(self, num_categories: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Input:
            category_ids: Tensor of shape (batch_size, seq_len)
        Output:
            category_embeddings: (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(category_ids)


class JointEmbedding(nn.Module):
    """
    Combines news + category embeddings into a single representation

    This module is shared by:
    - short-term user modeling
    - long-term user modeling
    """

    def __init__(
        self,
        num_news: int,
        num_categories: int,
        news_dim: int = 64,
        category_dim: int = 16
    ):
        super().__init__()

        self.news_embedding = NewsEmbedding(num_news, news_dim)
        self.category_embedding = CategoryEmbedding(num_categories, category_dim)

        self.output_dim = news_dim + category_dim

    def forward(
        self,
        news_ids: torch.Tensor,
        category_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            news_ids: (batch_size, seq_len)
            category_ids: (batch_size, seq_len)

        Output:
            joint_embeddings: (batch_size, seq_len, news_dim + category_dim)
        """

        news_vecs = self.news_embedding(news_ids)
        category_vecs = self.category_embedding(category_ids)

        # Concatenate along embedding dimension
        joint_embeddings = torch.cat([news_vecs, category_vecs], dim=-1)

        return joint_embeddings


# ----------------------------
# Sanity check (optional)
# ----------------------------
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5

    num_news = 100
    num_categories = 10

    model = JointEmbedding(
        num_news=num_news,
        num_categories=num_categories,
        news_dim=64,
        category_dim=16
    )

    news_ids = torch.randint(0, num_news, (batch_size, seq_len))
    category_ids = torch.randint(0, num_categories, (batch_size, seq_len))

    embeddings = model(news_ids, category_ids)

    print("Embedding shape:", embeddings.shape)
    # Expected: (2, 5, 80)
