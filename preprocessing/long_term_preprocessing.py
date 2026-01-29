from preprocessing.utils import chunk_interactions_by_day, build_daily_chunk_sequence
from preprocessing.sequence_builder import build_user_interaction_sequences

def prepare_long_term_input(user_interactions_with_dt):
    """
    Prepare long-term input for the model.
    Args:
        user_interactions_with_dt (dict): user_id -> [(news_id, timestamp, category, delta_t), ...]
    Returns:
        dict: user_id -> [(daily_interactions, delta_t_days),...]
    """
    
    # Partition interactions into daily chunks
    print("Chunking interactions by day...")
    user_daily_chunks = chunk_interactions_by_day(user_interactions_with_dt)

    # Order daily chunks and compute time gaps
    print("Building daily chunk sequences with day-level time gaps...")
    long_term_sequences = build_daily_chunk_sequence(user_daily_chunks)    

    return long_term_sequences

def run_long_term_preprocessing(user_interactions_with_dt):
    """
    End-to-end long-term preprocessing pipeline.
    Args:
        user_interactions_with_dt (dict): user_id -> [(news_id, timestamp, category, delta_t), ...]
    Returns:
        dict: user_id -> [(daily_interactions, delta_t_days), ...]
    """

    print("\n========== LONG-TERM PREPROCESSING START ==========\n")

    long_term_sequences = prepare_long_term_input(user_interactions_with_dt)

    print("========== LONG-TERM PREPROCESSING END ==========\n")

    return long_term_sequences

