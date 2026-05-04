import torch
import numpy as np
import pandas as pd
import path_variables as pv

from Logitha_LTC_Bert_Timeawarefusion import NewsRecModel, get_bert_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD DATA
# =========================
test_data = torch.load(pv.MIND_SMALL_PREPROCESSED_TEST, weights_only = False)
train_data = torch.load(pv.MIND_SMALL_PREPROCESSED_TRAIN, weights_only = False)

short_data = test_data["short_term_data"]

news2idx = train_data["news2idx"]
category2idx = train_data["category2idx"]

num_news = max(news2idx.values()) + 1
num_cat = max(category2idx.values()) + 1

# =========================
# INIT MODEL
# =========================
model = NewsRecModel(num_news, num_cat).to(device)
model.load_state_dict(torch.load("lohitha_model.pt", map_location=device))
print("Model weights loaded.\n")

# ⚠️ Her training code does NOT save model
# so we assume current model OR you modify train() to save

print("Model ready.\n")

# =========================
# BUILD CATEGORY MAP
# =========================
news_idx_to_cat = {}

for user in short_data:
    for x in short_data[user]:
        news_idx_to_cat[x[0]] = x[2]   # IMPORTANT: category index is x[2]

# =========================
# LOAD BEHAVIORS
# =========================
behaviors_df = pd.read_csv(
    pv.TEST_BEHAVIORS,
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"]
)

# =========================
# METRICS
# =========================
def compute_mrr(rank):
    return 0 if rank is None else 1.0 / rank

def compute_ndcg(rank, k):
    if rank is None or rank > k:
        return 0
    return 1.0 / np.log2(rank + 1)

def compute_auc(scores, clicked_index):
    pos = scores[clicked_index]
    neg = np.concatenate([scores[:clicked_index], scores[clicked_index+1:]])
    return np.mean(pos > neg)

# =========================
# EVALUATION
# =========================
model.eval()

mrr_list, ndcg5_list, ndcg10_list, auc_list = [], [], [], []

with torch.no_grad():
    for i, row in behaviors_df.iterrows():

        if i%1000 == 0:
            print(f"Processed {i} rows")

        user = row["user_id"]
        if user not in short_data:
            continue

        impressions = row["impressions"]
        if pd.isna(impressions):
            continue

        seq = short_data[user]
        if len(seq) < 2:
            continue

        input_seq = seq[:-1]

        news_seq = [x[0] for x in input_seq]
        cat_seq  = [x[2] for x in input_seq]
        delta_t  = [x[3] / 3600.0 for x in input_seq]

        news_seq = torch.tensor(news_seq).unsqueeze(0).to(device)
        cat_seq  = torch.tensor(cat_seq).unsqueeze(0).to(device)
        delta_t  = torch.tensor(delta_t).unsqueeze(0).to(device)

        content_vec = get_bert_batch(news_seq.cpu().numpy()).to(device)

        user_vec = model(news_seq, cat_seq, delta_t, content_vec)

        # parse candidates
        candidates = []
        clicked_index = None

        for idx, pair in enumerate(impressions.split(" ")):
            nid, label = pair.split("-")
            if nid not in news2idx:
                continue

            candidates.append(news2idx[nid])
            if label == "1":
                clicked_index = len(candidates) - 1

        if clicked_index is None or len(candidates) == 0:
            continue

        # compute scores
        scores = []
        for nid in candidates:
            nid_tensor = torch.tensor([[nid]]).to(device)
            cat_tensor = torch.tensor([[news_idx_to_cat.get(nid, 0)]]).to(device)

            content_vec = get_bert_batch(nid_tensor.cpu().numpy()).to(device)

            item_vec = model.encoder(nid_tensor, cat_tensor, content_vec).squeeze()

            score = torch.dot(user_vec.squeeze(), item_vec).item()
            scores.append(score)

        scores = np.array(scores)

        rank = np.argsort(-scores).tolist().index(clicked_index) + 1

        mrr_list.append(compute_mrr(rank))
        ndcg5_list.append(compute_ndcg(rank, 5))
        ndcg10_list.append(compute_ndcg(rank, 10))
        auc_list.append(compute_auc(scores, clicked_index))

# =========================
# PRINT RESULTS
# =========================
print("Valid samples:", len(mrr_list))
print("\n===== TEST METRICS =====")
print("MRR:", np.mean(mrr_list))
print("NDCG@5:", np.mean(ndcg5_list))
print("NDCG@10:", np.mean(ndcg10_list))
print("AUC:", np.mean(auc_list))