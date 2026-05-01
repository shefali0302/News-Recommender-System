"""
Trainable embedding layers
Responsible ONLY for mapping IDs -> dense vectors

Shared across:
- short-term modeling
- long-term modeling
"""

# This file defines the embedding layers for news articles and categories, as well as 
# a joint embedding that combines both. These embeddings are shared across the short-term 
# and long-term modeling components of the recommendation system.

import torch
import torch.nn as nn
import numpy as np


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
        # print("Max category id:", category_ids.max().item())
        # print("Min category id:", category_ids.min().item())

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
        category_dim: int = 16,
        idx2news = None
    ):
        super().__init__()
        self.news_dim = news_dim
        self.category_dim = category_dim
        self.idx2news = idx2news
        if self.idx2news is None:
            print("WARNING: idx2news not provided, BERT embeddings will fail")
        self.debug_done = False

        self.news_embedding = NewsEmbedding(num_news, news_dim)
        self.category_embedding = CategoryEmbedding(num_categories, category_dim)

        bert_path = "data/embeddings/news_bert_embeddings.npy" 
        self.bert_dict = np.load(bert_path, allow_pickle=True).item()  

        self.bert_dim = 384
        self.bert_proj_dim = 128 

        self.content_fc = nn.Linear(self.bert_dim, self.bert_proj_dim)

        self.output_dim = news_dim + category_dim + self.bert_proj_dim    



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

        batch_size, seq_len = news_ids.shape
        device = news_ids.device

        bert_vectors = []

        for i in range(batch_size):
            seq_vec = []
            for j in range(seq_len):
                idx = int(news_ids[i, j].item())


                if idx == 0:
                    vec = np.zeros(self.bert_dim)
                    news_id = None

                else:
                    if self.idx2news is not None:
                        news_id = self.idx2news.get(idx, None)
                    else:
                        news_id = None

                    if news_id is not None:
                        vec = self.bert_dict.get(news_id, np.zeros(self.bert_dim))
                    else:
                        vec = np.zeros(self.bert_dim)

                if not self.debug_done:
                    if i == 0 and j == 0:
                        print("\n===== DEBUG: EMBEDDING LOOKUP =====")
                        print("Index:", idx)
                        print("Mapped news_id:", news_id)
                        print("BERT vector sample:", vec[:5])
                        self.debug_done = True

                seq_vec.append(vec)
            bert_vectors.append(seq_vec)

        # bert_vectors = torch.tensor(
        #     bert_vectors,
        #     dtype = torch.float32,
        #     device = device
        # )
        bert_vectors = torch.from_numpy(
            np.array(bert_vectors)
        ).float().to(device)

        bert_proj = self.content_fc(bert_vectors)

        if not self.debug_done:  
            print("\n===== DEBUG: BERT PROJECTION =====")
            print("bert_vectors shape:", bert_vectors.shape)
            print("bert_proj shape:", bert_proj.shape)
            print("bert_proj variance:", bert_proj.var().item())
            self.debug_done = True

        joint_embeddings = torch.cat(
            [news_vecs, category_vecs, bert_proj],
            dim=-1
        )

        if not self.debug_done:
            print("\n===== DEBUG: FINAL EMBEDDING =====")
            print("news_vecs shape:", news_vecs.shape)
            print("category_vecs shape:", category_vecs.shape)
            print("bert_proj shape:", bert_proj.shape)
            print("joint_embeddings shape:", joint_embeddings.shape)
            self.debug_done = True

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
    # Expected: (2, 5, news_dim (128) + category_dim (128) + 128)
