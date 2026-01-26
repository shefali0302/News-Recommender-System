from models.long_term import LongTermModel
from preprocessing.long_term_preprocessing import prepare_long_term_input
from preprocessing.sequence_builder import compute_time_gaps
from preprocessing.dataset_ingestion import (
    load_news_categories,
    load_user_interactions,
    build_id_mappings,
    map_interactions_to_indices,
    sort_user_interactions
)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_PATH = os.path.join(BASE_DIR, "..", "data", "MINDsmall_train", "news.tsv")
BEHAVIORS_PATH = os.path.join(BASE_DIR, "..", "data", "MINDsmall_train", "behaviors.tsv")

# ---- Ingestion ----
news_cat_map = load_news_categories(NEWS_PATH)
user_interactions = load_user_interactions(BEHAVIORS_PATH, news_cat_map)

news2idx, cat2idx = build_id_mappings(user_interactions)
user_interactions = map_interactions_to_indices(user_interactions, news2idx, cat2idx)
user_interactions = sort_user_interactions(user_interactions)

# ---- Time gaps ----
user_interactions_with_dt = compute_time_gaps(user_interactions)

# ---- Long-term preprocessing ----
long_term_inputs = prepare_long_term_input(user_interactions_with_dt)

# ---- Pick one user ----
sample_user = next(iter(long_term_inputs))
long_term_sequence = long_term_inputs[sample_user]

print("Sample user:", sample_user)
print("Days:", len(long_term_sequence))

# ---- Model ----
model = LongTermModel(
    num_news=len(news2idx) + 1,
    num_categories=len(cat2idx) + 1
)

Z, delta_t = model(long_term_sequence)

print("Z shape:", Z.shape)
print("delta_t shape:", delta_t.shape)
print("First daily vector (first 10 values):")
print(Z[0][:10])
