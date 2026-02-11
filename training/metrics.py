import torch
import math


def compute_mrr(rank):
    if rank is None:
        return 0.0
    return 1.0 / rank


def compute_ndcg(rank, k):
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def compute_auc(scores, clicked_index):
    pos_score = scores[clicked_index]
    neg_scores = torch.cat([
        scores[:clicked_index],
        scores[clicked_index + 1:]
    ])

    if len(neg_scores) == 0:
        return 0.0

    return (pos_score > neg_scores).float().mean().item()
