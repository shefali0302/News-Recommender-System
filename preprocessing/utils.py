
import math
from typing import Counter
from collections import Counter
import numpy as np


#--------short term helper functions ---------

def get_last_n_interactions(user_interactions, N):
    user_last_n = {}

    for user_id, interactions in user_interactions.items():
        if len(interactions) >= N:
            user_last_n[user_id] = interactions[-N:]

    return user_last_n

def compute_dominant_categories(recent_interactions, alpha): #takes recent interactions of 1 user
    categories = [x[2] for x in recent_interactions]

    freq = Counter(categories)
    theta = math.ceil(alpha * len(recent_interactions))

    dominant_categories = [
        cat for cat, count in freq.items()
        if count >= theta
    ]

    return dominant_categories #returns list of dominant categories

def compute_time_thresholds(user_interactions_with_dt):
    dts = [
        dt for interactions in user_interactions_with_dt.values()
        for *_, dt in interactions if dt > 0
    ]

    dts = np.array(dts)
    return np.percentile(dts, 50), np.percentile(dts, 75)

def apply_time_mask(user_recent_interactions, tau):
    masked = {}

    for u, interactions in user_recent_interactions.items():
        masked[u] = [
            (nid, cat, dt, 1 if dt <= tau else 0)
            for nid, _, cat, dt in interactions
        ]

    return masked

def apply_category_mask(user_time_masked, user_dominant_categories):
    masked = {}

    for u, interactions in user_time_masked.items():
        dom = user_dominant_categories[u]
        masked[u] = [
            (*x, 1 if x[1] in dom else 0)
            for x in interactions
        ]

    return masked

def apply_hybrid_mask(user_category_masked):
    return {
        u: [
            (nid, cat, dt, m_time * m_cat)
            for nid, cat, dt, m_time, m_cat in interactions
        ]
        for u, interactions in user_category_masked.items()
    }
    return dominant_categories #returns number



#--------long term helper functions ---------
from collections import defaultdict

def chunk_interactions_by_day(user_interactions_with_dt):
    """
    Group user interactions by calendar day.

    Args:
        user_interactions_with_dt (dict): user_id -> [(news_id, timestamp, category, delta_t), ...]

    Returns:
        dict: user_id -> {date -> [(news_id, category, delta_t), ...]}
    """
    user_daily_chunks = {}

    for user_id, interactions in user_interactions_with_dt.items():
        daily_chunks = defaultdict(list)

        for news_id, timestamp, category, delta_t in interactions:
            day = timestamp.date()
            daily_chunks[day].append(
                (news_id, category, delta_t)
            )

        user_daily_chunks[user_id] = dict(daily_chunks)

    return user_daily_chunks

def build_daily_chunk_sequence(user_daily_chunks):
    """
    Build ordered daily interaction sequence with time gaps.

    Args:
        user_daily_chunks (dict):
            user_id -> {date -> [(news_id, category, delta_t), ...]}

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
