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
        device = user_vec.device

        candidate_ids = torch.tensor(
            candidate_news_ids,
            dtype=torch.long,
            device=device
        )  # (B, M)

        news_vecs = self.news_embedding(candidate_ids)  # (B, M, D)

        # batch matrix multiply
        scores = torch.bmm(news_vecs, user_vec.unsqueeze(-1)).squeeze(-1)  # (B, M)

        probs = torch.softmax(scores, dim=1)

        return scores, probs

