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


# if __name__ == "__main__":

#     #TASK 1
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#     NEWS_PATH = os.path.join(
#         BASE_DIR, "..", "data", "MINDsmall_train", "news.tsv"
#     )
#     BEHAVIORS_PATH = os.path.join(
#         BASE_DIR, "..", "data", "MINDsmall_train", "behaviors.tsv"
#     )

#     print("Loading news categories...")
#     news_category_map = load_news_categories(NEWS_PATH)

#     print("Loading user interactions...")
#     user_interactions = load_user_interactions(
#         BEHAVIORS_PATH,
#         news_category_map
#     )
#     #TASK 2
#     print("Sorting interactions by time...")
#     user_interactions = sort_user_interactions(user_interactions)

#     # Sanity check
#     sample_user = next(iter(user_interactions))
#     print(f"\nSample user: {sample_user}")
#     print(user_interactions[sample_user][:5])

    