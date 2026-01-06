"""
Long-term preprocessing orchestration.

This module prepares long-term input by:
1. Grouping interactions into daily chunks
2. Ordering daily chunks and computing day-level time gaps

- No embeddings
- No pooling
- No learning logic
"""

from preprocessing.utils import (
    chunk_interactions_by_day,
    build_daily_chunk_sequence
)

def prepare_long_term_input(user_interactions_with_dt):
    """
    Prepare long-term input for the model.

    Args:
        user_interactions_with_dt (dict):
            user_id -> [(news_id, timestamp, category, delta_t), ...]

    Returns:
        dict:
            user_id -> [(daily_interactions, delta_t_days),...]
    """
    # Partition interactions into daily chunks
    user_daily_chunks = chunk_interactions_by_day(user_interactions_with_dt)

    # Order daily chunks and compute time gaps
    long_term_sequences = build_daily_chunk_sequence(user_daily_chunks)

    return long_term_sequences
