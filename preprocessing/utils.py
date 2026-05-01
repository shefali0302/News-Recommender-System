
import math
from typing import Counter
from collections import Counter
import numpy as np
import torch


#--------short term helper functions ---------

def get_last_n_interactions(user_interactions, N):
    user_last_n = {}

    for user_id, interactions in user_interactions.items():
        if len(interactions) >= N:
            user_last_n[user_id] = interactions[-N:]

    return user_last_n

#--------long term helper functions ---------
from collections import defaultdict

def chunk_interactions_by_day(user_interactions_with_dt):
    """
    Group user interactions by calendar day.

    Args:
        user_interactions_with_dt (dict): user_id -> [(news_id, timestamp, cat_id, delta_t), ...]

    Returns:
        dict: user_id -> {date -> [(news_id, cat_id, delta_t), ...]}
    """
    user_daily_chunks = {}

    for user_id, interactions in user_interactions_with_dt.items():
        daily_chunks = defaultdict(list)

        for news_id, ts, cat_id, delta_t in interactions:
            day = ts.date()
            daily_chunks[day].append((news_id, ts, cat_id, delta_t))

        for day in daily_chunks:
            daily_chunks[day].sort(key=lambda x: x[1])  # sort by timestamp
            daily_chunks[day] = [(news_id, cat_id, delta_t) for news_id, ts, cat_id, delta_t in daily_chunks[day]]
        
        user_daily_chunks[user_id] = dict(daily_chunks)

    return user_daily_chunks

def build_daily_chunk_sequence(user_daily_chunks):
    """
    Build ordered daily interaction sequence with time gaps.

    Args:
        user_daily_chunks (dict):
            user_id -> {date -> [(news_id, cat_id, delta_t), ...]}

    Returns:
        dict:
            user_id -> [ (daily_interactions, delta_t_days), ...]
    """
    user_sequences = {}

    for user_id, daily_chunks in user_daily_chunks.items():
        sorted_days = sorted(daily_chunks.keys())
        sequence = []

        prev_day = None
        for day in sorted_days:
            delta_days = 0 if prev_day is None else (day - prev_day).days
            sequence.append((daily_chunks[day], delta_days))
            prev_day = day

        user_sequences[user_id] = sequence

    return user_sequences

def load_train_mappings(train_preprocessed_path):
    data = torch.load(train_preprocessed_path, weights_only=False)
    return data["news2idx"], data["category2idx"], data["idx2news"]

#for testing preprocessing pipeline
def map_to_existing_indices(user_interactions, news2idx, category2idx):

    mapped = {}

    for user_id, interactions in user_interactions.items():

        mapped_list = []

        for news_id, timestamp, category in interactions:

            news_idx = news2idx.get(news_id, 0)        # unseen → padding
            cat_idx  = category2idx.get(category, 0)

            mapped_list.append((news_idx, timestamp, cat_idx))

        # sort chronologically
        mapped_list.sort(key=lambda x: x[1])

        # compute delta_t
        with_dt = []
        prev_time = None

        for news_idx, timestamp, cat_idx in mapped_list:

            if prev_time is None:
                delta_t = 0
            else:
                delta_t = (timestamp - prev_time).total_seconds()

            with_dt.append(
                (news_idx, timestamp, cat_idx, delta_t)
            )

            prev_time = timestamp

        mapped[user_id] = with_dt

    return mapped
