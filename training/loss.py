import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

def compute_loss(scores, clicked_index):
    """
    scores: tensor (M,)
    clicked_index: int
    """

    pos_score = scores[clicked_index]

    # all negatives
    neg_scores = torch.cat([scores[:clicked_index], scores[clicked_index + 1:]])

    # BPRmax loss
    loss = -torch.log(torch.sigmoid(pos_score - neg_scores) + 1e-8).mean()

    return loss
