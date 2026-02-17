# ----------------------------------------------------------------------
# This is the main training script for the news recommendation system.
# It loads the preprocessed data, initializes the models, and runs the training loop.
# After training, it also runs a simple evaluation to check the performance of the model.
# ----------------------------------------------------------------------
import os
import torch
import torch.optim as optim
from models.scoring import ItemScorer
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC
from models.fusion import FusionGate
from models.embeddings import JointEmbedding
from training.loss import compute_loss
from training.metrics import compute_mrr, compute_ndcg, compute_auc
from training.experiment_util import create_experiment_folder
from path_variables import DATASET, TRAIN_NEWS, TRAIN_BEHAVIORS


MODE = "full"
# options: "full", "short_only", "long_only", "no_gate"
BATCH_SIZE = 32
NUM_EPOCHS = 15


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_data(path):
    return torch.load(path)


def evaluate_model(short_model, long_model, fusion_gate, scorer,
                   short_term_data, long_term_data, MODE):

    print("Evaluating model...")
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

            # scores, _ = scorer(user_vec, [candidates])
            # scores = scores[0]

            # Make user_vec batch-shaped
            user_vec = user_vec.unsqueeze(0)  # (1, D)
            scores, _ = scorer(user_vec, [candidates])
            scores = scores.squeeze(0)  # (M,)


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

def train_model(train_short, train_long,
                val_short, val_long,
                news2idx, category2idx,
                num_epochs):

    num_news = max(news2idx.values()) + 1
    num_categories = max(category2idx.values()) + 1

    joint_embedding = JointEmbedding(num_news, num_categories, 64).to(device)
    short_model = ShortTermLTC(joint_embedding, hidden_dim=64).to(device)
    long_model = LongTermLTC(joint_embedding, hidden_dim=64).to(device)
    fusion_gate = FusionGate(dim=64).to(device)
    scorer = ItemScorer(joint_embedding).to(device)

    optimizer = optim.Adam(
        list(short_model.parameters()) +
        list(long_model.parameters()) +
        list(fusion_gate.parameters()) +
        list(scorer.parameters()),
        lr=0.001
    )

    best_score = 0
    patience = 3
    no_improve = 0

    user_ids = list(train_short.keys())

    for epoch in range(num_epochs):
        print("epoch: ", epoch+1)
        print("short term model training")
        short_model.train()
        print("long term model training")
        long_model.train()
        print("fusion gate training")
        fusion_gate.train()
        print("item scorer training")    
        scorer.train()

        total_loss = 0

        for batch_start in range(0, len(user_ids), BATCH_SIZE):
            print("starting batch: ", batch_start)

            batch_users = user_ids[batch_start: batch_start + BATCH_SIZE]

            batch_st = []
            batch_lt = []
            batch_candidates = []
            batch_targets = []

            for user_id in batch_users:

                if user_id not in train_long:
                    continue

                short_seq = train_short[user_id]
                long_seq = train_long[user_id]

                if len(short_seq) < 2:
                    continue

                candidates = [x[0] for x in short_seq]
                clicked_index = len(candidates) - 1

                st_vec, _, _ = short_model(short_seq)
                lt_vec, _, _ = long_model(long_seq)

                batch_st.append(st_vec)
                batch_lt.append(lt_vec)

                batch_candidates.append(candidates)
                batch_targets.append(clicked_index)

            if len(batch_st) == 0:
                continue

            # batch_st = torch.cat(batch_st, dim=0)
            # batch_lt = torch.cat(batch_lt, dim=0)

            batch_st = torch.stack(batch_st, dim=0)   # (B, D)
            batch_lt = torch.stack(batch_lt, dim=0)   # (B, D)


            print("batch_st shape:", batch_st.shape, "epoch: ", epoch+1)

            if MODE == "short_only":
                user_vec = batch_st
            elif MODE == "long_only":
                user_vec = batch_lt
            elif MODE == "no_gate":
                user_vec = 0.5 * (batch_st + batch_lt)
            else:
                user_vec, _ = fusion_gate(batch_st, batch_lt)

            scores, _ = scorer(user_vec, batch_candidates)

            loss = 0
            for i in range(len(batch_targets)):
                loss += compute_loss(scores[i], batch_targets[i])

            loss = loss / len(batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch {} Loss: {:.4f}".format(epoch+1, total_loss))

        val_metrics = evaluate_model(
            short_model, long_model,
            fusion_gate, scorer,
            val_short, val_long, MODE
        )

        print("Validation:", val_metrics)

        current_score = val_metrics["NDCG@10"]

        if current_score > best_score:
            best_score = current_score
            no_improve = 0

            torch.save({
                "joint_embedding": joint_embedding.state_dict(),
                "short_model": short_model.state_dict(),
                "long_model": long_model.state_dict(),
                "fusion_gate": fusion_gate.state_dict(),
                "scorer": scorer.state_dict()
            }, "best_model.pt")

            print("New best model saved.")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete")

    return short_model, long_model, fusion_gate, scorer

if __name__ == "__main__":

    MODE = "full"   # change this for ablation runs
    # options: "full", "short_only", "long_only", "no_gate"

    train_data = load_data("data/MINDsmall_train_preprocessed_train.pt")
    dev_data = load_data("data/MINDsmall_train_preprocessed_val.pt")

    if DATASET == "MINDlarge":
        train_data = load_data("data/MINDlarge_train_preprocessed.pt")
        dev_data = load_data("data/MINDlarge_dev_preprocessed.pt")

    train_short = train_data["short_term_data"]
    train_long = train_data["long_term_data"]
    news2idx = train_data["news2idx"]
    category2idx = train_data["category2idx"]

    dev_short = dev_data["short_term_data"]
    dev_long = dev_data["long_term_data"]

    config = {
        "embedding_dim": 64,
        "hidden_dim": 64,
        "learning_rate": 0.001,
        "num_epochs": NUM_EPOCHS,
        "mode": MODE,
        "optimizer": "Adam",
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "dataset": "MINDsmall"
    }

    exp_dir = create_experiment_folder(config)
    
    short_model, long_model, fusion_gate, scorer = train_model(
        train_short, train_long,
        dev_short, dev_long,
        news2idx, category2idx,
        num_epochs=NUM_EPOCHS
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

    with open(os.path.join(exp_dir, "final_metrics.txt"), "w") as f:
        for k, v in test_metrics.items():
            f.write("{}: {:.4f}\n".format(k, v))
