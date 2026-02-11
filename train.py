# ----------------------------------------------------------------------
# This is the main training script for the news recommendation system.
# It loads the preprocessed data, initializes the models, and runs the training loop.
# After training, it also runs a simple evaluation to check the performance of the model.
# ----------------------------------------------------------------------

import torch
import torch.optim as optim
from preprocessing.run_preprocessing_pipeline import run_preprocessing_pipeline
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC
from models.fusion import FusionGate
from models.scoring import ItemScorer
from training.loss import compute_loss
from inference.top_k_recs import get_top_k_news
from training.metrics import compute_mrr, compute_ndcg, compute_auc


MODE = "full"
# options: "full", "short_only", "long_only", "no_gate"

MAX_USERS=20

def evaluate_model(short_model, long_model, fusion_gate, scorer,
                   short_term_data, long_term_data, MODE):

    mrr_list = []
    ndcg5_list = []
    ndcg10_list = []
    auc_list = []

    for i, user_id in enumerate(short_term_data):

        if i >= MAX_USERS:
            break

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

        # ---- ablation switch ----
        if MODE == "short_only":
            user_vec = st_vec
        elif MODE == "long_only":
            user_vec = lt_vec
        elif MODE == "no_gate":
            user_vec = 0.5 * (st_vec + lt_vec)
        else:  # full model
            user_vec, _ = fusion_gate(st_vec, lt_vec)

        scores, _ = scorer(user_vec, candidates)

        # ---- ranking ----
        sorted_indices = torch.argsort(scores, descending=True)
        rank_pos = (sorted_indices == clicked_index).nonzero(as_tuple=True)

        if len(rank_pos[0]) == 0:
            rank = None
        else:
            rank = rank_pos[0].item() + 1  # 1-based rank

        # ---- metrics ----
        mrr_list.append(compute_mrr(rank))
        ndcg5_list.append(compute_ndcg(rank, 5))
        ndcg10_list.append(compute_ndcg(rank, 10))
        auc_list.append(compute_auc(scores, clicked_index))

    return {
        "MRR": sum(mrr_list) / len(mrr_list),
        "NDCG@5": sum(ndcg5_list) / len(ndcg5_list),
        "NDCG@10": sum(ndcg10_list) / len(ndcg10_list),
        "AUC": sum(auc_list) / len(auc_list)
    }

def train_model(num_epochs):
    short_term_data, long_term_data, news2idx, category2idx = run_preprocessing_pipeline()

    num_news = max(news2idx.values()) + 1
    num_categories = max(category2idx.values()) + 1

    short_model = ShortTermLTC(num_news, num_categories, hidden_dim=64)
    long_model  = LongTermLTC(num_news, num_categories, hidden_dim=64)
    fusion_gate = FusionGate(dim=64)
    scorer      = ItemScorer(num_news=num_news, embedding_dim=64)

    optimizer = optim.Adam(
        list(short_model.parameters()) +
        list(long_model.parameters()) +
        list(fusion_gate.parameters()) +
        list(scorer.parameters()),
        lr=0.001
    )

    for epoch in range(num_epochs):

        total_loss = 0

        for i, user_id in enumerate(short_term_data):

            if i >= MAX_USERS:
                break

            if user_id not in long_term_data:
                continue

            short_seq = short_term_data[user_id]
            long_seq  = long_term_data[user_id]

            if len(short_seq) < 2:
                continue

            # candidates = news in short-term window
            candidates = [x[0] for x in short_seq]

            # target = last clicked item
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


            scores, probs = scorer(user_vec, candidates)

            loss = compute_loss(scores, clicked_index)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss = {total_loss:.4f}")

    print("Training complete")

    user_id = next(iter(short_term_data))

    st_vec, _, _ = short_model(short_term_data[user_id])
    lt_vec, _, _ = long_model(long_term_data[user_id])

    if MODE == "short_only":
        user_vec = st_vec
    elif MODE == "long_only":
        user_vec = lt_vec
    elif MODE == "no_gate":
        user_vec = 0.5 * (st_vec + lt_vec)
    else:
        user_vec, _ = fusion_gate(st_vec, lt_vec)

    candidates = [x[0] for x in short_term_data[user_id]]
    scores, probs = scorer(user_vec, candidates)

    top_k = get_top_k_news(scores, candidates, k=5)

    print("Top K:", top_k)

    return short_model, long_model, fusion_gate, scorer, short_term_data, long_term_data



if __name__ == "__main__":

    MODE = "no_gate"   # change this for ablation runs
    # options: "full", "short_only", "long_only", "no_gate"

    short_model, long_model, fusion_gate, scorer, short_term_data, long_term_data = train_model(num_epochs=5)


    print("\nRunning evaluation...")
    metrics = evaluate_model(
        short_model,
        long_model,
        fusion_gate,
        scorer,
        short_term_data,
        long_term_data,
        MODE
    )

    print("\nEvaluation Results")
    print(f"Mode: {MODE}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


    
