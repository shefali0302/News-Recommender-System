
from preprocessing.dataset_ingestion import (
    load_news_categories,
    load_user_interactions,
    build_id_mappings,
    map_interactions_to_indices,
    sort_user_interactions
)
from preprocessing.long_term_preprocessing import prepare_long_term_input
from preprocessing.sequence_builder import compute_time_gaps


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_PATH = os.path.join(BASE_DIR, "..", "data", "MINDsmall_train", "news.tsv")
BEHAVIORS_PATH = os.path.join(BASE_DIR, "..", "data", "MINDsmall_train", "behaviors.tsv")

news_cat_map = load_news_categories(NEWS_PATH)
user_interactions = load_user_interactions(BEHAVIORS_PATH, news_cat_map)

news2idx, cat2idx = build_id_mappings(user_interactions)
user_interactions = map_interactions_to_indices(user_interactions, news2idx, cat2idx)
user_interactions = sort_user_interactions(user_interactions)
user_interactions_with_dt = compute_time_gaps(user_interactions)

long_term_inputs = prepare_long_term_input(user_interactions_with_dt)

sample_user = next(iter(long_term_inputs))

print("Sample user:", sample_user)
print("Number of days:", len(long_term_inputs[sample_user]))

daily_interactions, delta_days = long_term_inputs[sample_user][0]

print("\nFirst day interactions:")
print(daily_interactions[:3])
print("Delta days:", delta_days)
