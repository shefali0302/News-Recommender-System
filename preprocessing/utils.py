
import math
from typing import Counter
from collections import Counter
import numpy as np

N=10
alpha=0.5


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

    return dominant_categories #returns list of dominant categories

def compute_dominant_categories_all_users(user_recent_interactions, alpha):
    """
    Input:
      dict {user_id: [(news_id, ts, category, dt), ...]}

    Output:
      dict {user_id: set(dominant_categories)}
    """
    result = {}

    for user_id, interactions in user_recent_interactions.items():
        result[user_id] = set(
            compute_dominant_categories(interactions, alpha)
        )

    return result

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