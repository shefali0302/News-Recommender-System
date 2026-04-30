"""
This module parses the MIND dataset and converts raw behavior logs into time-ordered, user-wise interaction sequences with category information.
"""

import os
import pandas as pd
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm



def load_news_categories(news_path):
    """
    Returns: dict {news_id: category}
    """
    news_df = pd.read_csv(
        news_path,
        sep="\t",
        header=None,
        names=[
            "news_id", "category", "subcategory",
            "title", "abstract", "url",
            "title_entities", "abstract_entities"
        ]
    )

    news_category_map = dict(
        zip(news_df["news_id"], news_df["category"])
    )

    return news_category_map

def load_user_interactions(behaviors_path, news_category_map):
    """
    Returns:
    dict {user_id: [(news_id, timestamp, category), ...]}
    """

    behaviors_df = pd.read_csv(
        behaviors_path,
        sep="\t",
        header=None,
        names=[
            "impression_id", "user_id",
            "time", "history", "impressions"
        ],
        #dtype=str   
    )


    user_interactions = defaultdict(list)

    for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df)):
        user_id = row["user_id"]
        time_str = row["time"]

        # Convert timestamp string to datetime object
        timestamp = datetime.strptime(
            time_str, "%m/%d/%Y %I:%M:%S %p"
        )

        history = row["history"]
        # impressions = row["impressions"]

        news_ids = []

        # Add history clicks
        if not pd.isna(history):
            news_ids.extend(history.split(" "))

        # Add impression candidates
        # if not pd.isna(impressions):
        #     impression_pairs = impressions.split(" ")
        #     for pair in impression_pairs:
        #         news_id = pair.split("-")[0]
        #         news_ids.append(news_id)

        for news_id in news_ids:
            category = news_category_map.get(news_id, "unknown")
            user_interactions[user_id].append(
                (news_id, timestamp, category)
            )

    return user_interactions

def build_id_mappings(news_category_map):
    """
    Build integer ID mappings for news and categories.
    Padding index 0 is reserved.
    """

    news_ids = list(news_category_map.keys())
    categories = list(set(news_category_map.values()))

    news2idx = {nid: i + 1 for i, nid in enumerate(news_ids)}
    idx2news = {i + 1: nid for i, nid in enumerate(news_ids)}
    cat2idx  = {cat: i + 1 for i, cat in enumerate(categories)}

    return news2idx, cat2idx, idx2news

def map_interactions_to_indices(user_interactions, news2idx, cat2idx):
    """
    Convert (news_id, timestamp, category)
    → (news_idx, timestamp, cat_idx)
    """
    mapped = {}

    for user_id, interactions in user_interactions.items():
        mapped[user_id] = [
            (news2idx[news_id], timestamp, cat2idx[category])
            for news_id, timestamp, category in interactions
        ]

    return mapped

def sort_user_interactions(user_interactions):
    for user_id in user_interactions:
        user_interactions[user_id].sort(
            key=lambda x: x[1]  # sort by timestamp
        )
    return user_interactions

