import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

def compute_loss(scores, clicked_index):
    """
    scores: tensor (M,)
    clicked_index: int
    """
    target = torch.tensor([clicked_index])
    scores = scores.unsqueeze(0)
    loss = loss_fn(scores, target)
    return loss
