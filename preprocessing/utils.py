
import math
from typing import Counter
import numpy

N=10
alpha=0.5

#--------short term helper functions ---------

def get_last_n_interactions(user_interactions, N):
    user_last_n = {}

    for user_id, interactions in user_interactions.items():
        if len(interactions) >= N:
            user_last_n[user_id] = interactions[-N:]

    return user_last_n

def compute_dominant_categories(recent_interactions, alpha): #takes recent inteactions of 1 user
    categories = [x[2] for x in recent_interactions]

    freq = Counter(categories)
    theta = math.ceil(alpha * len(recent_interactions))

    dominant_categories = [
        cat for cat, count in freq.items()
        if count >= theta
    ]

    return dominant_categories #returns number



#--------long term helper functions ---------
from collections import defaultdict

def chunk_interactions_by_day(user_interactions):
    """
    Group user interactions by calendar day.

    Args:
        user_interactions (dict): user_id -> [(news_id, timestamp, category, delta_t), ...]

    Returns:
        dict: user_id -> {date -> [(news_id, category, delta_t), ...]}
    """
    user_daily_chunks = {}

    for user_id, interactions in user_interactions.items():
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
