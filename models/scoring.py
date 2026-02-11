# ----------------------------------------------------------------------------------
# This file defines the scoring component of the recommendation system, which takes 
# the fused user representation and candidate news items to produce relevance scores 
# and probabilities.
# ----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch
import torch.nn as nn

class ItemScorer(nn.Module):

    def __init__(self, num_news, embedding_dim=64):
        super().__init__()
        self.news_embedding = nn.Embedding(num_news, embedding_dim)

    def forward(self, user_vec, candidate_news_ids):
        """
        user_vec: (D,)
        candidate_news_ids: list[int]
        """

        candidate_ids = torch.tensor(candidate_news_ids)
        news_vecs = self.news_embedding(candidate_ids)   # (M, D)

        scores = torch.matmul(news_vecs, user_vec)       # (M,)

        probs = torch.softmax(scores, dim=0)

        return scores, probs
