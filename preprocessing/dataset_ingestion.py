"""
This module parses the MIND dataset and converts raw behavior logs into time-ordered, user-wise interaction sequences with category information.
"""

import os
import pandas as pd
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NEWS_PATH = os.path.join(
        BASE_DIR, "..", "data", "MINDsmall_train", "news.tsv"
    )
BEHAVIORS_PATH = os.path.join(
        BASE_DIR, "..", "data", "MINDsmall_train", "behaviors.tsv"
    )

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
        ]
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

        if pd.isna(history):
            continue

        clicked_news = history.split(" ")

        for news_id in clicked_news:
            category = news_category_map.get(news_id, "unknown")
            user_interactions[user_id].append(
                (news_id, timestamp, category)
            )

    return user_interactions

def sort_user_interactions(user_interactions):
    for user_id in user_interactions:
        user_interactions[user_id].sort(
            key=lambda x: x[1]  # sort by timestamp
        )
    return user_interactions

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

            enriched_interactions.append(
                (news_id, timestamp, category, delta_t)
            )

            prev_time = timestamp

        user_interactions_with_dt[user_id] = enriched_interactions

    return user_interactions_with_dt

def extract_sliding_window(user_interactions_with_dt, N):
    """
    Input:
      dict {user_id: [(news_id, timestamp, category, delta_t), ...]}

    Output:
      dict {user_id: last N interactions}
    """

    user_recent_interactions = {}

    for user_id, interactions in user_interactions_with_dt.items():
        if len(interactions) <= N:
            user_recent_interactions[user_id] = interactions
        else:
            user_recent_interactions[user_id] = interactions[-N:]

    return user_recent_interactions


import math
from collections import Counter

def detect_dominant_categories(user_recent_interactions, N, alpha):
    """
    Input:
      dict {user_id: [(news_id, timestamp, category, delta_t), ...]}

    Output:
      dict {user_id: set of dominant categories}
    """

    user_dominant_categories = {}

    for user_id, interactions in user_recent_interactions.items():
        categories = [item[2] for item in interactions]

        category_counts = Counter(categories)

        theta = math.ceil(alpha * len(interactions))

        dominant = [
            cat for cat, count in category_counts.items()
            if count >= theta
        ]

        if not dominant:
            # fallback- most frequent category
            dominant = [category_counts.most_common(1)[0][0]]

        user_dominant_categories[user_id] = set(dominant)

    return user_dominant_categories

def build_user_interactions():
    news_category_map = load_news_categories(NEWS_PATH)
    user_interactions = load_user_interactions(
        BEHAVIORS_PATH, news_category_map
    )
    user_interactions = sort_user_interactions(user_interactions)
    user_interactions = compute_time_gaps(user_interactions)
    return user_interactions


if __name__ == "__main__":
    

    #TASK 1
    

    print("Loading news categories...")
    news_category_map = load_news_categories(NEWS_PATH)

    print("Loading user interactions...")
    user_interactions = load_user_interactions(
        BEHAVIORS_PATH,
        news_category_map
    )
    #TASK 2
    print("Sorting interactions by time...")
    user_interactions = sort_user_interactions(user_interactions)

    # Sanity check
    sample_user = next(iter(user_interactions))
    print(f"\nSample user: {sample_user}")
    print(user_interactions[sample_user][:5])

    #TASK 3
    print("Computing time gaps (Δt)...")
    user_interactions_with_dt = compute_time_gaps(user_interactions)

    # Sanity check
    sample_user = next(iter(user_interactions_with_dt))
    print(f"\nSample user with Δt: {sample_user}")
    print(user_interactions_with_dt[sample_user][:5])

    #TASK 4
    N = 10
    print(f"Extracting last {N} interactions per user...")
    user_recent_interactions = extract_sliding_window(
        user_interactions_with_dt, N
    )

    # Sanity check
    sample_user = next(iter(user_recent_interactions))
    print(f"\nSample user recent interactions (N={N}): {sample_user}")
    print(user_recent_interactions[sample_user])

    #TASK 5
    alpha = 0.4
    print("Detecting dominant categories...")
    user_dominant_categories = detect_dominant_categories(
        user_recent_interactions, N, alpha
    )

    # Sanity check
    sample_user = next(iter(user_dominant_categories))
    print(f"\nSample user's dominant categories: {sample_user}")
    print(user_dominant_categories[sample_user])

    user_interactions=build_user_interactions()



