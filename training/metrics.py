# --------------------------------------------------------------
# This file contains functions to compute evaluation metrics 
# like MRR, NDCG, and AUC for the recommendation model.
# --------------------------------------------------------------

import torch
import math


# This function computes the MRR metric for a given rank.
# MRR (Mean Reciprocal Rank) measures the average of the reciprocal ranks of the relevant items.
def compute_mrr(rank):
    if rank is None:
        return 0.0
    return 1.0 / rank

# This function computes the NDCG metric for a given rank and cutoff k.
# NDCG is a measure of ranking quality that takes into account the position of the relevant item.
def compute_ndcg(rank, k):
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


# This function computes the AUC metric for a given set of scores and the index of the clicked item.
# AUC (Area Under the Curve) measures the ability of the model to rank a randomly chosen positive instance higher than a randomly chosen negative instance.
def compute_auc(scores, clicked_index):
    pos_score = scores[clicked_index]
    neg_scores = torch.cat([
        scores[:clicked_index],
        scores[clicked_index + 1:]
    ])

    if len(neg_scores) == 0:
        return 0.0

    return (pos_score > neg_scores).float().mean().item()
