# ----------------------------------------------------------------------------------
# This file defines the scoring component of the recommendation system, which takes 
# the fused user representation and candidate news items to produce relevance scores 
# and probabilities.
# ----------------------------------------------------------------------------------

import torch
import torch.nn as nn

class ItemScorer(nn.Module):
    def __init__(self, joint_embedding):
        super().__init__()
        self.news_embedding = joint_embedding.news_embedding

    def forward(self, user_vec, candidate_news_ids):
        """
        user_vec: (B, D) or (B, 1, D)
        candidate_news_ids: list of lists (len B)
        """

        device = user_vec.device
        batch_size = len(candidate_news_ids)

        # ---- Fix user_vec shape ----
        if user_vec.dim() == 3:
            user_vec = user_vec.squeeze(1)   # (B, D)

        # ---- Pad candidate ids ----
        max_len = max(len(c) for c in candidate_news_ids)

        padded_candidates = torch.zeros(
            batch_size, max_len,
            dtype=torch.long,
            device=device
        )

        for i, cand in enumerate(candidate_news_ids):
            padded_candidates[i, :len(cand)] = torch.tensor(
                cand,
                dtype=torch.long,
                device=device
            )

        # ---- Get news embeddings ----
        news_vecs = self.news_embedding(padded_candidates)  # (B, M, D)

        # ---- Ensure proper dims for bmm ----
        user_vec = user_vec.unsqueeze(-1)  # (B, D, 1)

        scores = torch.bmm(news_vecs, user_vec).squeeze(-1)  # (B, M)

        return scores, torch.softmax(scores, dim=1)


