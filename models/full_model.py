import torch
import torch.nn as nn

from short_term import ShortTermPipeline
from long_term import LongTermPipeline
from fusion import FusionGate
from scoring import ItemScoring


class FullRecommender(nn.Module):

    def __init__(self,
                 num_news,
                 num_categories,
                 news_dim=64,
                 category_dim=16,
                 hidden_dim=64):

        super().__init__()

        # Short-term
        self.short_pipeline = ShortTermPipeline(
            num_news, num_categories,
            news_dim, category_dim, hidden_dim
        )

        # Long-term
        self.long_pipeline = LongTermPipeline(
            num_news, num_categories,
            news_dim, category_dim, hidden_dim
        )

        # Fusion
        self.fusion = FusionGate(hidden_dim)

        # IMPORTANT: share embedding layer with scoring
        self.item_scorer = ItemScoring(
            self.short_pipeline.short_term_model.embedding_layer
        )

    def forward(self,
                short_seq,
                long_seq,
                candidate_news_ids,
                candidate_cat_ids):

        st_vec, _, _ = self.short_pipeline(short_seq)
        lt_vec, _, _ = self.long_pipeline(long_seq)

        st_vec = st_vec.squeeze(0)
        lt_vec = lt_vec.squeeze(0)

        fused_user_vec, gate = self.fusion(st_vec, lt_vec)

        scores, probs = self.item_scorer(
            fused_user_vec,
            candidate_news_ids,
            candidate_cat_ids
        )

        return scores, probs, gate
