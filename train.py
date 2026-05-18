# ---------------------------------------------------------------------------------------
# This is the main training script for the news recommendation system.
# It loads the preprocessed data, initializes the models, and runs the training loop.
# After training, it also runs a simple evaluation to check the performance of the model.
# ---------------------------------------------------------------------------------------
import os
import torch
import random
import torch.optim as optim
import path_variables as pv
import pandas as pd
import numpy as np
from models.scoring import ItemScorer
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC
from models.fusion import FusionGate
from models.embeddings import JointEmbedding
from training.loss import compute_loss
from training.metrics import compute_mrr, compute_ndcg, compute_auc
from training.experiment_util import create_experiment_folder
from path_variables import DATASET, TRAIN_NEWS, TRAIN_BEHAVIORS, DEV_BEHAVIORS


MODE = "full"
# options: "full", "short_only", "long_only", "no_gate"
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 0.001

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_data(path):
    return torch.load(path)


def evaluate_model(short_model, long_model, fusion_gate, scorer,
                   short_term_data, long_term_data, MODE,
                   behaviors_path, news2idx, news_idx_to_cat):

    behaviors_df = pd.read_csv(
        behaviors_path,
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"]
    )

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

        for _, row in behaviors_df.iterrows():

            user_id = row["user_id"]

            if user_id not in short_term_data: continue

            impressions = row["impressions"]

            if pd.isna(impressions): continue

            short_seq = short_term_data[user_id]
            long_seq = long_term_data.get(user_id, None)

            if long_seq is None or len(short_seq) < 2:
                continue

            impression_pairs = impressions.split(" ")

            candidates = []
            clicked_index = None

            for idx, pair in enumerate(impression_pairs):
                news_id, label = pair.split("-")

                if news_id not in news2idx:
                    continue

                candidates.append(news2idx[news_id])

                if label == "1":
                    clicked_index = len(candidates) - 1

            if clicked_index is None or len(candidates) == 0:
                continue

            # Remove clicked news from history to avoid leakage
            clicked_news_idx = candidates[clicked_index]

            short_seq = [
                x for x in short_seq
                if x[0] != clicked_news_idx
            ]

            if len(short_seq) < 1:
                continue

            # ===== Compute user representation =====
            st_vec, _, _ = short_model(short_seq)
            lt_vec, _, _ = long_model(long_seq)

            if st_vec is None or lt_vec is None:
                continue

            st_vec = st_vec.squeeze(0)
            lt_vec = lt_vec.squeeze(0)

            if MODE == "short_only":
                user_vec = st_vec
            elif MODE == "long_only":
                user_vec = lt_vec
            elif MODE == "no_gate":
                user_vec = 0.5 * (st_vec + lt_vec)
            else:
                user_vec, _ = fusion_gate(st_vec, lt_vec)

            # Batch shape
            user_vec = user_vec.unsqueeze(0)

            # Score candidates
            # build category list
            candidate_cats = []

            for n in candidates:
                # fallback safe lookup
                if n in news_idx_to_cat:
                    candidate_cats.append(news_idx_to_cat[n])
                else:
                    candidate_cats.append(0)  # padding safety

            scores, _ = scorer(
                user_vec,
                [candidates],
                [candidate_cats]
            )
            scores = scores.squeeze(0)

            # Rank computation
            sorted_indices = torch.argsort(scores, descending=True)
            rank_pos = (sorted_indices == clicked_index).nonzero(as_tuple=True)

            rank = rank_pos[0].item() + 1 if len(rank_pos[0]) > 0 else None

            # Metrics
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
                news2idx, category2idx, idx2news,
                num_epochs, news_idx_to_cat, cat_to_news, impression_negative_map):

    num_news = max(news2idx.values()) + 1
    num_categories = max(category2idx.values()) + 1
    news_dim = 128
    category_dim = 32

    joint_embedding = JointEmbedding(num_news, num_categories, news_dim = news_dim, category_dim = category_dim, idx2news=idx2news).to(device)
    short_model = ShortTermLTC(joint_embedding, hidden_dim=256).to(device)
    long_model = LongTermLTC(joint_embedding, hidden_dim=256).to(device)
    fusion_gate = FusionGate(dim=256).to(device)
    scorer = ItemScorer(joint_embedding).to(device)

    optimizer = optim.Adam(
        # list(joint_embedding.parameters())+
        list(short_model.parameters()) +
        list(long_model.parameters()) +
        list(fusion_gate.parameters()) +
        list(scorer.parameters()),
        lr=LR
    )


    best_score = -float("inf")
    patience = 5
    no_improve = 0

    user_ids = list(train_short.keys())

    all_news_indices = list(range(1, max(news2idx.values()) + 1))
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
            batch_candidate_categories = []
            batch_targets = []

            for user_id in batch_users:

                if user_id not in train_long:
                    continue

                short_seq = train_short[user_id]
                long_seq = train_long[user_id]

                if len(short_seq) < 2:
                    continue

                # -----------------------------
                # Leave-One-Out Setup
                # -----------------------------
                input_seq = short_seq[:-1]     # remove last item
                positive_item = short_seq[-1][0]   # news_idx

                # Build user history set to avoid sampling clicked items
                user_clicked = set(x[0] for x in short_seq)

                # -----------------------------
                # Negative Sampling
                # -----------------------------

                K = 6
                POOL_SIZE = 30

                IMPRESSION_RATIO = 0.7
                CATEGORY_RATIO = 0.3

                num_impression = int(POOL_SIZE * IMPRESSION_RATIO)
                num_category = int(POOL_SIZE * CATEGORY_RATIO)

                candidate_pool = []

                # --- 1. same-category negatives ---
                pos_cat = news_idx_to_cat.get(positive_item, 0)
                
                same_cat_pool = cat_to_news.get(pos_cat, [])

                same_cat_negs = []
                for neg in same_cat_pool:
                    if neg not in user_clicked and neg != positive_item:
                        same_cat_negs.append(neg)
                    if len(same_cat_negs) >= num_category:
                        break

                # --- 2. random negatives ---
                # random_negs = []
                # while len(random_negs) < POOL_SIZE // 2:
                #     neg = random.choice(all_news_indices)
                #     if neg not in user_clicked:
                #         random_negs.append(neg)

                # --- 2. impression negatives ---

                impression_negs = impression_negative_map.get(
                    positive_item,
                    []
                )

                filtered_impression_negs = []

                for neg in impression_negs:

                    if neg not in user_clicked and neg != positive_item:
                        filtered_impression_negs.append(neg)

                # sample from impression negatives
                if len(filtered_impression_negs) > num_impression:
                    impression_negs_sampled = random.sample(
                        filtered_impression_negs,
                        num_impression
                    )
                else:
                    impression_negs_sampled = filtered_impression_negs
                
                candidate_pool = same_cat_negs + impression_negs_sampled

                while len(candidate_pool) < POOL_SIZE:

                    neg = random.choice(all_news_indices)

                    if neg not in user_clicked and neg != positive_item:
                        candidate_pool.append(neg)

                candidate_pool = list(set(candidate_pool))  
                random.shuffle(candidate_pool)
                                
                with torch.no_grad():
                    st_vec_tmp, _, _ = short_model(input_seq)
                    lt_vec_tmp, _, _ = long_model(long_seq)

                    if st_vec_tmp is None or lt_vec_tmp is None:
                        continue

                    st_vec_tmp = st_vec_tmp.squeeze(0)
                    lt_vec_tmp = lt_vec_tmp.squeeze(0)

                    if MODE == "short_only":
                        user_vec_tmp = st_vec_tmp
                    elif MODE == "long_only":
                        user_vec_tmp = lt_vec_tmp
                    elif MODE == "no_gate":
                        user_vec_tmp = 0.5 * (st_vec_tmp + lt_vec_tmp)
                    else:
                        user_vec_tmp, _ = fusion_gate(st_vec_tmp, lt_vec_tmp)

                    user_vec_tmp = user_vec_tmp.unsqueeze(0)

                    candidate_category_pool = [news_idx_to_cat.get(idx, 0) for idx in candidate_pool]
                    scores_pool, _ = scorer(user_vec_tmp, [candidate_pool], [candidate_category_pool])
                    scores_pool = scores_pool.squeeze(0)

                top_k = min(K, len(candidate_pool));
                _, top_indices = torch.topk(scores_pool, top_k)

                negatives = [candidate_pool[i] for i in top_indices.tolist()]
                
                # Candidate set = 1 positive + K negatives
                candidates = [positive_item] + negatives
                random.shuffle(candidates)

                clicked_index = candidates.index(positive_item)
                candidate_categories = [news_idx_to_cat.get(n, 0) for n in candidates]

                # -----------------------------
                # Compute user representation
                # -----------------------------
                st_vec, _, _ = short_model(input_seq)
                lt_vec, _, _ = long_model(long_seq)

                if st_vec is None or lt_vec is None:
                    continue

                st_vec = st_vec.squeeze(0)
                lt_vec = lt_vec.squeeze(0)

                batch_st.append(st_vec)
                batch_lt.append(lt_vec)
                batch_candidates.append(candidates)
                batch_candidate_categories.append(candidate_categories)
                batch_targets.append(clicked_index)

            if len(batch_st) == 0:
                continue

            batch_st = torch.stack(batch_st, dim=0)   # (B, D)
            batch_lt = torch.stack(batch_lt, dim=0)   # (B, D)

            if MODE == "short_only":
                user_vec = batch_st
            elif MODE == "long_only":
                user_vec = batch_lt
            elif MODE == "no_gate":
                user_vec = 0.5 * (batch_st + batch_lt)
            else:
                user_vec, _ = fusion_gate(batch_st, batch_lt)

            # user_vec = user_vec.squeeze(1)

            scores, _ = scorer(user_vec, batch_candidates, batch_candidate_categories)

            loss = torch.stack([
                compute_loss(scores[i], batch_targets[i])
                for i in range(len(batch_targets))
            ]).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                # list(joint_embedding.parameters()) +
                list(short_model.parameters()) +
                list(long_model.parameters()) +
                list(fusion_gate.parameters()) +
                list(scorer.parameters()),
                5.0
            )
            optimizer.step()

            total_loss += loss.item()

        print("Epoch {} Loss: {:.4f}".format(epoch+1, total_loss))

        val_metrics = evaluate_model(
            short_model, long_model,
            fusion_gate, scorer,
            val_short, val_long, MODE,
            pv.DEV_BEHAVIORS,
            news2idx,
            news_idx_to_cat
        )

        print("Validation:", val_metrics)

        current_score = val_metrics["NDCG@10"]

        if current_score > best_score + 1e-4:
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

    train_data = load_data(pv.MIND_SMALL_PREPROCESSED_TRAIN )
    dev_data = load_data(pv.MIND_SMALL_PREPROCESSED_DEV)

    #if DATASET == "MINDlarge":
        #train_data = load_data(pv.MIND_LARGE_PREPROCESSED_TRAIN)
        #dev_data = load_data(pv.MIND_LARGE_PREPROCESSED_DEV)

    train_short = train_data["short_term_data"]
    train_long = train_data["long_term_data"]
    news2idx = train_data["news2idx"]
    category2idx = train_data["category2idx"]
    idx2news = train_data["idx2news"]
    impression_negative_map = train_data["impression_negative_map"]

    dev_short = dev_data["short_term_data"]
    dev_long = dev_data["long_term_data"]

    config = {
        "embedding_dim": 128,
        "hidden_dim": 256,
        "learning_rate": LR,
        "num_epochs": NUM_EPOCHS,
        "mode": MODE,
        "optimizer": "Adam",
        "batch_size": BATCH_SIZE,
        "device": str(device),
        "dataset": "MINDsmall"
    }

    exp_dir = create_experiment_folder(config)

    # Build news_idx → category_idx mapping
    news_idx_to_cat = {}

    for user in train_short:
        for x in train_short[user]:
            news_idx = x[0]
            cat_idx = x[1]
            news_idx_to_cat[news_idx] = cat_idx

    for user in train_long:
        for day, _ in train_long[user]:
            for x in day:
                news_idx = x[0]
                cat_idx = x[1]
                news_idx_to_cat[news_idx] = cat_idx
    
    from collections import defaultdict
    cat_to_news = defaultdict(list)

    for n, c in news_idx_to_cat.items():
        cat_to_news[c].append(n)

    short_model, long_model, fusion_gate, scorer = train_model(
        train_short, train_long,
        dev_short, dev_long,
        news2idx, category2idx, idx2news,
        num_epochs=NUM_EPOCHS,
        news_idx_to_cat = news_idx_to_cat,
        cat_to_news = cat_to_news,
        impression_negative_map=impression_negative_map
    )

    print("\nFinal Evaluation on Dev Set")

   

    test_metrics = evaluate_model(
        short_model,
        long_model,
        fusion_gate,
        scorer,
        dev_short,
        dev_long,
        MODE,
        pv.DEV_BEHAVIORS,
        news2idx,
        news_idx_to_cat
    )

    with open(os.path.join(exp_dir, "final_metrics.txt"), "w") as f:
        for k, v in test_metrics.items():
            f.write("{}: {:.4f}\n".format(k, v))
