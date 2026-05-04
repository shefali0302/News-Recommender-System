from Logitha_LTC_Bert_Timeawarefusion import NewsRecModel, train
import torch
import path_variables as pv

from torch.utils.data import DataLoader
from data_loader_lohitha import SimpleDataset, collate_fn

# =========================
# LOAD DATA
# =========================
data = torch.load(pv.MIND_SMALL_PREPROCESSED_TRAIN)

short_data = data["short_term_data"]
news2idx = data["news2idx"]
category2idx = data["category2idx"]

# =========================
# DATASET + DATALOADER
# =========================
dataset = SimpleDataset(short_data, news2idx)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# =========================
# MODEL INIT
# =========================
num_news = max(news2idx.values()) + 1
num_cat = max(category2idx.values()) + 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NewsRecModel(num_news, num_cat).to(device)

# =========================
# TRAIN
# =========================
train(model, dataloader)