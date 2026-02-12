# ----------------------------------------------------------------------
# This is the main training script for the news recommendation system.
# It loads the preprocessed data, initializes the models, and runs the training loop.
# After training, it also runs a simple evaluation to check the performance of the model.
# ----------------------------------------------------------------------

import torch
import torch.optim as optim
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC
from models.fusion import FusionGate
from models.scoring import ItemScorer
from training.loss import compute_loss
from inference.top_k_recs import get_top_k_news
from training.metrics import compute_mrr, compute_ndcg, compute_auc


MODE = "full"
# options: "full", "short_only", "long_only", "no_gate"

MAX_USERS=100

def load_data(path):
    return torch.load(path)


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

        for i, user_id in enumerate(short_term_data):

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
        "MRR": sum(mrr_list) / max(1, len(mrr_list)),
        "NDCG@5": sum(ndcg5_list) / max(1, len(ndcg5_list)),
        "NDCG@10": sum(ndcg10_list) / max(1, len(ndcg10_list)),
        "AUC": sum(auc_list) / max(1, len(auc_list))
    }

def train_model(train_short, train_long, val_short, val_long, news2idx, category2idx, num_epochs):
    best_score = 0
    best_epoch = 0
    patience = 3
    no_improve = 0
    

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
        short_model.train()
        long_model.train()
        fusion_gate.train()
        scorer.train()

        total_loss = 0

        for i, user_id in enumerate(train_short):

            if user_id not in train_long:
                continue

            short_seq = train_short[user_id]
            long_seq  = train_long[user_id]

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

        val_metrics = evaluate_model(
            short_model,
            long_model,
            fusion_gate,
            scorer,
            val_short,
            val_long,
            MODE
        )

        current_score = val_metrics["NDCG@10"]
        print(f"Validation NDCG@10: {current_score:.4f}")

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            no_improve = 0

            torch.save({
                "short_model": short_model.state_dict(),
                "long_model": long_model.state_dict(),
                "fusion_gate": fusion_gate.state_dict(),
                "scorer": scorer.state_dict()
            }, "best_model.pt")

            print(f"New best model saved at epoch {best_epoch}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break


    print("Training complete")
    print(f"Loading best model from epoch {best_epoch}")
    checkpoint = torch.load("best_model.pt")
    short_model.load_state_dict(checkpoint["short_model"])
    long_model.load_state_dict(checkpoint["long_model"])
    fusion_gate.load_state_dict(checkpoint["fusion_gate"])
    scorer.load_state_dict(checkpoint["scorer"])

    return short_model, long_model, fusion_gate, scorer



if __name__ == "__main__":

    MODE = "full"   # change this for ablation runs
    # options: "full", "short_only", "long_only", "no_gate"

    train_data = load_data("data/MINDsmall_train_preprocessed_train.pt")
    dev_data = load_data("data/MINDsmall_train_preprocessed_val.pt")

    train_short = train_data["short_term_data"]
    train_long = train_data["long_term_data"]
    news2idx = train_data["news2idx"]
    category2idx = train_data["category2idx"]

    dev_short = dev_data["short_term_data"]
    dev_long = dev_data["long_term_data"]

    short_model, long_model, fusion_gate, scorer = train_model(
        train_short, train_long,
        dev_short, dev_long,
        news2idx, category2idx,
        num_epochs=20
    )

    print("\nFinal Evaluation on Dev Set")

    test_metrics = evaluate_model(
        short_model,
        long_model,
        fusion_gate,
        scorer,
        dev_short,
        dev_long,
        MODE
    )

    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
