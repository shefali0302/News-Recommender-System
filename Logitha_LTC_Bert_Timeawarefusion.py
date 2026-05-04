#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:57:22 2026

@author: Logitha
"""
#!/usr/bin/env python3
# ==========================================
# FINAL MODEL: LTC + BERT + TIME-AWARE FUSION
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM = 128
HIDDEN_DIM = 128
BERT_DIM = 384
LR = 1e-3
EPOCHS = 10
SHORT_N = 10

# ==========================================
# LOAD PRECOMPUTED BERT EMBEDDINGS
# ==========================================
# bert_dict = np.load("news_bert_embeddings.npy", allow_pickle=True).item()
bert_dict = np.load("data/embeddings/news_bert_embeddings.npy", allow_pickle=True).item()

def get_bert_batch(news_ids):
    batch = []
    for seq in news_ids:
        seq_vec = []
        for nid in seq:
            vec = bert_dict.get(int(nid), np.zeros(BERT_DIM))
            seq_vec.append(vec)
        batch.append(seq_vec)
    return torch.from_numpy(np.array(batch)).float()


# ==========================================
# ENCODER (NOW WITH BERT)
# ==========================================
class NewsEncoder(nn.Module):
    def __init__(self, num_news, num_cat):
        super().__init__()

        self.news_emb = nn.Embedding(num_news, EMBED_DIM)
        self.cat_emb = nn.Embedding(num_cat, EMBED_DIM)

        self.content_fc = nn.Linear(BERT_DIM, EMBED_DIM)

    def forward(self, news_id, category, content_vec):
        n = self.news_emb(news_id)
        c = self.cat_emb(category)
        content = self.content_fc(content_vec)

        return torch.cat([n, c, content], dim=-1)


# ==========================================
# TRUE LTC
# ==========================================
class LTCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.tau = nn.Parameter(torch.ones(hidden_dim))

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, x, delta_t):
        B, N, D = x.shape

        h = torch.zeros(B, self.hidden_dim).to(x.device)
        outputs = []

        for t in range(N):
            xt = x[:, t, :]
            dt = delta_t[:, t].unsqueeze(-1)

            x_proj = self.input_proj(xt)
            g = self.gate(torch.cat([h, xt], dim=-1))

            decay = torch.exp(-dt / (self.tau + 1e-6))
            h = decay * h + (1 - decay) * (g * x_proj)

            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)


# ==========================================
# ATTENTION
# ==========================================
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, 1)

    def forward(self, H):
        w = torch.softmax(self.W(H), dim=1)
        return torch.sum(w * H, dim=1)


# ==========================================
# FUSION GATE (IMPROVED)
# ==========================================
class FusionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, st, lt):
        interaction = st * lt
        g = self.fc(torch.cat([st, lt, interaction], dim=-1))
        return g * st + (1 - g) * lt


# ==========================================
# FULL MODEL
# ==========================================
class NewsRecModel(nn.Module):
    def __init__(self, num_news, num_cat):
        super().__init__()

        self.encoder = NewsEncoder(num_news, num_cat)

        input_dim = EMBED_DIM * 3

        self.ltc_short = LTCEncoder(input_dim, HIDDEN_DIM)
        self.ltc_long = LTCEncoder(input_dim, HIDDEN_DIM)

        self.attn_s = Attention(HIDDEN_DIM)
        self.attn_l = Attention(HIDDEN_DIM)

        self.fusion = FusionGate(HIDDEN_DIM)

    def forward(self, news_seq, cat_seq, delta_t, content_vec):

        x = self.encoder(news_seq, cat_seq, content_vec)

        # Soft time weighting
        time_weight = torch.exp(-delta_t).unsqueeze(-1)
        x = x * time_weight

        # Split
        short_x = x[:, -SHORT_N:, :]
        short_dt = delta_t[:, -SHORT_N:]

        long_x = x
        long_dt = delta_t

        # LTC
        st = self.attn_s(self.ltc_short(short_x, short_dt))
        lt = self.attn_l(self.ltc_long(long_x, long_dt))

        return self.fusion(st, lt)


# ==========================================
# LOSS
# ==========================================
def bpr_loss(user_vec, pos_vec, neg_vec):
    pos = torch.sum(user_vec * pos_vec, dim=-1)
    neg = torch.sum(user_vec * neg_vec, dim=-1)
    return -torch.mean(torch.log(torch.sigmoid(pos - neg)))


# ==========================================
# TRAIN
# ==========================================
def train(model, dataloader):

    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in dataloader:
            news_seq, cat_seq, delta_t, pos_item, neg_item = batch

            news_seq = news_seq.to(device)
            cat_seq = cat_seq.to(device)
            delta_t = delta_t.to(device)

            # 🔥 BERT batch
            content_vec = get_bert_batch(news_seq.cpu().numpy()).to(device)

            pos_vec = model.encoder.news_emb(pos_item.to(device))
            neg_vec = model.encoder.news_emb(neg_item.to(device))

            optimizer.zero_grad()

            user_vec = model(news_seq, cat_seq, delta_t, content_vec)

            loss = bpr_loss(user_vec, pos_vec, neg_vec)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    
    torch.save(model.state_dict(), "lohitha_model.pt")
    print("✅ Training Done")