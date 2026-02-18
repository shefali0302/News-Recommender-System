
import torch
import os
import path_variables as pv

from models.embeddings import JointEmbedding
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC
from models.fusion import FusionGate
from models.scoring import ItemScorer

from training.metrics import compute_mrr, compute_ndcg, compute_auc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def evaluate_model(short_model, long_model, fusion_gate, scorer,
                   short_term_data, long_term_data, MODE):

    short_model.eval()
    long_model.eval()
    fusion_gate.eval()
    scorer.eval()

    mrr_list = []
    ndcg5_list = []
    ndcg10_list = []
    auc_list = []

    with torch.no_grad():

        for user_id in short_term_data:

            if user_id not in long_term_data:
                continue

            short_seq = short_term_data[user_id]
            long_seq = long_term_data[user_id]

            if len(short_seq) < 2:
                continue

            candidates = [x[0] for x in short_seq]
            clicked_index = len(candidates) - 1

            st_vec, _, _ = short_model(short_seq)
            lt_vec, _, _ = long_model(long_seq)

            if MODE == "short_only":
                user_vec = st_vec
            elif MODE == "long_only":
                user_vec = lt_vec
            elif MODE == "no_gate":
                user_vec = 0.5 * (st_vec + lt_vec)
            else:
                user_vec, _ = fusion_gate(st_vec, lt_vec)

            scores, _ = scorer(user_vec, candidates)

            
            user_vec = user_vec.unsqueeze(0)  # (1, D)
            scores, _ = scorer(user_vec, [candidates])
            scores = scores.squeeze(0) 

            sorted_indices = torch.argsort(scores, descending=True)
            rank_pos = (sorted_indices == clicked_index).nonzero(as_tuple=True)

            rank = rank_pos[0].item() + 1 if len(rank_pos[0]) > 0 else None

            mrr_list.append(compute_mrr(rank))
            ndcg5_list.append(compute_ndcg(rank, 5))
            ndcg10_list.append(compute_ndcg(rank, 10))
            auc_list.append(compute_auc(scores, clicked_index))

    return {
        "MRR": sum(mrr_list) / max(1, len(mrr_list)),
        "NDCG@5": sum(ndcg5_list) / max(1, len(ndcg5_list)),
        "NDCG@10": sum(ndcg10_list) / max(1, len(ndcg10_list)),
        "AUC": sum(auc_list) / max(1, len(auc_list))
    }

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

    checkpoint = torch.load("results/best_model.pt", map_location=device)

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
        MODE="full"
    )

    print("========== TEST METRICS ==========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
