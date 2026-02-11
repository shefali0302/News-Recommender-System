# --------------------------------------------------------------------------------------
# This file defines the function to get the top-k recommended news items based on the
# scores produced by the scoring component.
# --------------------------------------------------------------------------------------

import torch

def get_top_k_news(scores, candidate_ids, k=5):

    topk = torch.topk(scores, k=min(k, len(scores)))

    top_indices = topk.indices.tolist()

    top_news_ids = [candidate_ids[i] for i in top_indices]

    return top_news_ids