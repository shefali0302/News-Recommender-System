#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:54:55 2026

@author: Logita
"""
# ==========================================
# BERT EMBEDDING GENERATION (RUN ONCE)
# ==========================================

from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
    return emb.squeeze().cpu().numpy()

# Load MIND news.tsv
news_df = pd.read_csv("data/MINDsmall/train/news.tsv", sep="\t", header=None)
news_df.columns = ["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]

embeddings = {}

for _, row in tqdm(news_df.iterrows(), total=len(news_df)):
    text = str(row["title"]) + " " + str(row["abstract"])
    embeddings[row["id"]] = get_embedding(text)

np.save("data/embeddings/news_bert_embeddings.npy", embeddings)