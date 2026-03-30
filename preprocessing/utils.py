
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

def compute_dominant_categories(recent_interactions, alpha): 
    categories = [cat_id for _, _, cat_id, _ in recent_interactions]

    freq = Counter(categories)
    theta = math.ceil(alpha * len(recent_interactions))

    dominant_categories = [
        cat for cat, count in freq.items()
        if count >= theta
    ]
    if len(dominant_categories) == 0:
        dominant_categories = [max(freq, key=freq.get)]  # fallback to most frequent category

    return dominant_categories #returns list of dominant categories

def compute_time_thresholds(user_interactions_with_dt):
    dts = [
        dt for interactions in user_interactions_with_dt.values()
        for *_, dt in interactions if dt > 0
    ]

    dts = np.array(dts)
    return np.percentile(dts, 50), np.percentile(dts, 75)

def apply_time_mask(user_recent_interactions, tau):
    time_masked_user_recent_interactions = {}

    for user_id, interactions in user_recent_interactions.items():
        time_masked_user_recent_interactions[user_id] = [
            (news_id, ts, cat_id, dt, 1 if dt <= tau else 0)
            for news_id, ts, cat_id, dt in interactions
        ]

    return time_masked_user_recent_interactions

def apply_category_mask(time_masked_user_recent_interactions, user_dominant_categories):
    cat_masked_user_recent_interactions = {}

    for user_id, interactions in time_masked_user_recent_interactions.items():
        dom = user_dominant_categories[user_id]
        cat_masked_user_recent_interactions[user_id] = [
            (news_id, ts, cat_id, dt, m_time, 1 if cat_id in dom else 0)
            for news_id, ts, cat_id, dt, m_time in interactions
        ]

    return cat_masked_user_recent_interactions

def apply_hybrid_mask(user_category_masked):
    return {
        user_id: [
            (news_id, ts, cat_id, dt, m_time * m_cat)
            for news_id, ts, cat_id, dt, m_time, m_cat in interactions
        ]
        for user_id, interactions in user_category_masked.items()
    }



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
    return data["news2idx"], data["category2idx"]

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
