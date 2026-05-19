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
        self.joint_embedding = joint_embedding
        self.user_proj = nn.Linear(256, 256)
        # self.scorer = nn.Sequential(
        #     nn.Linear(256 * 3, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 1)
        # )

    def forward(self, user_vec, candidate_news_ids, candidate_cat_ids):
        """
        user_vec: (B, D) or (B, 1, D)
        candidate_news_ids: list of lists (len B)
        candidate_cat_ids: list of lists (len B)
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
        padded_categories = torch.zeros(
            batch_size, max_len,
            dtype=torch.long,
            device=device
        )

        for i, (candidate_news, candidate_category) in enumerate(zip(candidate_news_ids, candidate_cat_ids)):
            padded_candidates[i, :len(candidate_news)] = torch.tensor(
                candidate_news,
                dtype=torch.long,
                device=device
            )
            padded_categories[i, :len(candidate_category)] = torch.tensor(
                candidate_category,
                dtype=torch.long,
                device=device
            )

        # ---- Get news embeddings ----
        news_vecs = self.joint_embedding(padded_candidates, padded_categories)  # (B, M, D)

        # ---- Ensure proper dims for bmm ----
        user_vec = self.user_proj(user_vec) # (B, D)

        # ---- MLP scoring layer ----
        # user_expand = user_vec.unsqueeze(1).expand(-1, news_vecs.size(1), -1)  # (B, M, D)

        # interaction = user_expand * news_vecs  # (B, M, D)
        # combined = torch.cat([user_expand, news_vecs, interaction], dim=-1)

        # scores = self.scorer(combined).squeeze(-1)  # (B, M)

        # ---- Dot product scoring layer ----
        user_vec = user_vec.unsqueeze(-1)  # (B, D, 1)

        scores = torch.bmm(
            news_vecs,
            user_vec
        ).squeeze(-1)  # (B, M)

        return scores, torch.softmax(scores, dim=1)
        
 