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
    def __init__(self, joint_embedding):
        super().__init__()
        self.news_embedding = joint_embedding.news_embedding

    def forward(self, user_vec, candidate_news_ids):
        """
        user_vec: (D,)
        candidate_news_ids: list[int]
        """
        device = user_vec.device

        candidate_ids = torch.tensor(candidate_news_ids, dtype=torch.long, device=device)  # (M,)
        news_vecs = self.news_embedding(candidate_ids)   # (M, D)

        scores = torch.matmul(news_vecs, user_vec)       # (M,)

        probs = torch.softmax(scores, dim=0)

        return scores, probs
