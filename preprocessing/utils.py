
import math
from typing import Counter
import numpy

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

    return dominant_categories #returns number
