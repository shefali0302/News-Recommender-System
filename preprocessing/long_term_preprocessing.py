"""
Long-term preference modeling:
Daily chunking, mean pooling, and time-aware sequence construction.
"""

from collections import defaultdict
import numpy as np


def chunk_interactions_by_day(user_interaction_embeddings):
    """
    Group interaction embeddings by calendar day.

    Args:
        user_interaction_embeddings (dict):
            user_id → [(timestamp, x_i), ...]

    Returns:
        dict:
            user_id → {date → [x_i, x_j, ...]}
    """
    user_daily_chunks = {}

    for user_id, interactions in user_interaction_embeddings.items():
        daily_chunks = defaultdict(list)

        for timestamp, embedding in interactions:
            day = timestamp.date()
            daily_chunks[day].append(embedding)

        user_daily_chunks[user_id] = dict(daily_chunks)

    return user_daily_chunks


def mean_pool_daily_chunks(user_daily_chunks):
    """
    Mean pool interaction embeddings within each daily chunk.

    Args:
        user_daily_chunks (dict):
            user_id → {date → [x_i, x_j, ...]}

    Returns:
        dict:
            user_id → {date → z_m (daily preference vector)}
    """
    user_daily_vectors = {}

    for user_id, daily_chunks in user_daily_chunks.items():
        daily_vectors = {}

        for day, embeddings in daily_chunks.items():
            embeddings = np.stack(embeddings)   # shape: (N_m, d)
            daily_vectors[day] = embeddings.mean(axis=0)

        user_daily_vectors[user_id] = daily_vectors

    return user_daily_vectors


def build_daily_sequence_with_time_gaps(user_daily_vectors):
    """
    Build ordered daily preference sequence with time gaps.

    Args:
        user_daily_vectors (dict):
            user_id → {date → z_m}

    Returns:
        dict:
            user_id → [(z_m, delta_t_days), ...]
    """
    user_sequences = {}

    for user_id, daily_vectors in user_daily_vectors.items():
        sorted_days = sorted(daily_vectors.keys())
        sequence = []

        prev_day = None
        for day in sorted_days:
            delta_t = 0 if prev_day is None else (day - prev_day).days
            sequence.append((daily_vectors[day], delta_t))
            prev_day = day

        user_sequences[user_id] = sequence

    return user_sequences
