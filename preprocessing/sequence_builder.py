"""
Preprocessing orchestration layer.
Builds user interaction sequences with time gaps.
"""

import os
from preprocessing.dataset_ingestion import (
    build_id_mappings,
    load_news_categories,
    load_user_interactions,
    map_interactions_to_indices,
    sort_user_interactions
)
from path_variables import TRAIN_NEWS, TRAIN_BEHAVIORS

def compute_time_gaps(user_interactions):
    """
    Input:
      dict {user_id: [(news_id, timestamp, category), ...]}

    Output:
      dict {user_id: [(news_id, timestamp, category, delta_t), ...]}
      where delta_t is time gap in seconds
    """

    user_interactions_with_dt = {}

    for user_id, interactions in user_interactions.items():
        enriched_interactions = []

        prev_time = None

        for news_id, timestamp, category in interactions:
            if prev_time is None:
                delta_t = 0.0  # first interaction
            else:
                delta_t = (timestamp - prev_time).total_seconds()

            enriched_interactions.append((news_id, timestamp, category, delta_t))

            prev_time = timestamp

        user_interactions_with_dt[user_id] = enriched_interactions

    return user_interactions_with_dt


def build_user_interaction_sequences():
    """

    Returns:
        dict:
            user_id -> [(news_id, timestamp, category, delta_t),...]
    """

    print("\n========== SEQUENCE CONSTRUCTION START ==========\n")

    news_category_map = load_news_categories(TRAIN_NEWS)

    user_interactions = load_user_interactions(TRAIN_BEHAVIORS, news_category_map)
    
    news2idx, cat2idx = build_id_mappings(news_category_map)

    user_interactions = map_interactions_to_indices(user_interactions, news2idx, cat2idx)

    user_interactions = sort_user_interactions(user_interactions)

    user_interactions_with_dt = compute_time_gaps(user_interactions)

    print("\n========== SEQUENCE CONSTRUCTION END ==========\n")

    print("\n===== DEBUG: SAMPLE USER INTERACTIONS =====")

    for i, (user_id, interactions) in enumerate(user_interactions_with_dt.items()):
        print("\nUser:", user_id)
        print("Num interactions:", len(interactions))

        for x in interactions[:5]:
            print(x)

        if i >= 2:
            break

    return user_interactions_with_dt, news2idx, cat2idx
