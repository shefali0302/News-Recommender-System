"""
Preprocessing orchestration layer.
Builds user interaction sequences with time gaps.
"""

import os
from preprocessing.dataset_ingestion import (
    load_news_categories,
    load_user_interactions,
    sort_user_interactions
)


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


def build_user_interaction_sequences(data_dir):
    """
    Args:
        data_dir (str): path to MIND dataset directory

    Returns:
        dict:
            user_id -> [(news_id, timestamp, category, delta_t),...]
    """

    news_path = os.path.join(data_dir, "news.tsv")
    behaviors_path = os.path.join(data_dir, "behaviors.tsv")

    news_category_map = load_news_categories(news_path)

    user_interactions = load_user_interactions(behaviors_path, news_category_map)

    user_interactions = sort_user_interactions(user_interactions)

    user_interactions_with_dt = compute_time_gaps(user_interactions)

    return user_interactions_with_dt
