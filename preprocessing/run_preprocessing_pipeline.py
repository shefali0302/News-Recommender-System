from preprocessing.sequence_builder import build_user_interaction_sequences
from preprocessing.short_term_preprocessing import run_short_term_preprocessing
from preprocessing.long_term_preprocessing import run_long_term_preprocessing
from preprocessing.configs import N, alpha


def run_preprocessing_pipeline():
    # Build base interaction sequences
    user_interactions_with_dt, news2idx, category2idx = build_user_interaction_sequences()

    # Short-term
    short_term_data = run_short_term_preprocessing(N, alpha, user_interactions_with_dt)

    # Long-term
    long_term_data = run_long_term_preprocessing(user_interactions_with_dt)


    return short_term_data, long_term_data, news2idx, category2idx
