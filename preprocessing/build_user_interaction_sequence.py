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
from preprocessing.time_gap import compute_time_gaps


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

    user_interactions = load_user_interactions(
        behaviors_path, news_category_map
    )

    user_interactions = sort_user_interactions(user_interactions)

    user_interactions_with_dt = compute_time_gaps(
        user_interactions
    )

    return user_interactions_with_dt
