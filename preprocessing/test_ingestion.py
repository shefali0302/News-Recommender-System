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

print("Loading news categories...")
news_cat_map = load_news_categories(NEWS_PATH)

print("Loading user interactions...")
user_interactions = load_user_interactions(BEHAVIORS_PATH, news_cat_map)

print("Building ID mappings...")
news2idx, cat2idx = build_id_mappings(user_interactions)

print("Mapping interactions to indices...")
user_interactions = map_interactions_to_indices(
    user_interactions, news2idx, cat2idx
)

print("Sorting interactions...")
user_interactions = sort_user_interactions(user_interactions)

# Pick ONE user
sample_user = next(iter(user_interactions))

print("\nSample user:", sample_user)
print("First 5 interactions:")
print(user_interactions[sample_user][:5])

print("\nSanity checks:")
news_idx, ts, cat_idx = user_interactions[sample_user][0]
print("news_idx type:", type(news_idx))
print("category_idx type:", type(cat_idx))
