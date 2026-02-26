
import torch
import os
import path_variables as pv

from models.embeddings import JointEmbedding
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC
from models.fusion import FusionGate
from models.scoring import ItemScorer
from train import evaluate_model
from training.metrics import compute_mrr, compute_ndcg, compute_auc
from path_variables import DATASET, TEST_NEWS, TEST_BEHAVIORS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    test_data = torch.load(pv.MIND_SMALL_PREPROCESSED_TEST)

    short_term_data = test_data["short_term_data"]
    long_term_data = test_data["long_term_data"]

    train_data = torch.load(pv.MIND_SMALL_PREPROCESSED_TRAIN)

    news2idx = train_data["news2idx"]
    category2idx = train_data["category2idx"]

    num_news = max(news2idx.values()) + 1
    num_categories = max(category2idx.values()) + 1

    joint_embedding = JointEmbedding(num_news, num_categories, 64).to(device)
    short_model = ShortTermLTC(joint_embedding, hidden_dim=64).to(device)
    long_model = LongTermLTC(joint_embedding, hidden_dim=64).to(device)
    fusion_gate = FusionGate(dim=64).to(device)
    scorer = ItemScorer(joint_embedding).to(device)

    checkpoint = torch.load("best_model.pt", map_location=device)

    joint_embedding.load_state_dict(checkpoint["joint_embedding"])
    short_model.load_state_dict(checkpoint["short_model"])
    long_model.load_state_dict(checkpoint["long_model"])
    fusion_gate.load_state_dict(checkpoint["fusion_gate"])
    scorer.load_state_dict(checkpoint["scorer"])

    print("Model loaded successfully.\n")

    metrics = evaluate_model(
        short_model,
        long_model,
        fusion_gate,
        scorer,
        short_term_data,
        long_term_data,
        MODE="full",
        behaviors_path=pv.TEST_BEHAVIORS,
        news2idx=news2idx
    )

    print("========== TEST METRICS ==========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
