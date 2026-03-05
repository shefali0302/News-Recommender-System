import torch
import os
from preprocessing.dataset_ingestion import load_news_categories, load_user_interactions
from preprocessing.sequence_builder import build_user_interaction_sequences
from preprocessing.short_term_preprocessing import run_short_term_preprocessing
from preprocessing.long_term_preprocessing import run_long_term_preprocessing
from preprocessing.configs import N, alpha
import path_variables as pv
import preprocessing.utils as utils


def run_preprocessing_pipeline():
    # Build base interaction sequences
    user_interactions_with_dt, news2idx, category2idx = build_user_interaction_sequences()

    # Short-term
    short_term_data = run_short_term_preprocessing(N, alpha, user_interactions_with_dt)

    # Long-term
    long_term_data = run_long_term_preprocessing(user_interactions_with_dt)


    return short_term_data, long_term_data, news2idx, category2idx

def run_test_preprocessing_pipeline():
    news2idx, category2idx = utils.load_train_mappings(os.path.join("data", "preprocessed", f"preprocessed_{pv.DATASET}_train"))

    news_category_map = load_news_categories(pv.TEST_NEWS)
    raw_interactions = load_user_interactions(
        pv.TEST_BEHAVIORS,
        news_category_map
    )

    user_interactions_with_dt = utils.map_to_existing_indices(
        raw_interactions,
        news2idx,
        category2idx
    )
    short_term_data = run_short_term_preprocessing(
        N,
        alpha,
        user_interactions_with_dt
    )

    long_term_data = run_long_term_preprocessing(
        user_interactions_with_dt
    )

    return short_term_data, long_term_data


if __name__ == "__main__":
    print("\n========== PREPROCESSING START ==========\n")

    save_path = os.path.join("data", "preprocessed", f"preprocessed_{pv.DATASET}_{pv.MODE}.pt")
    if pv.MODE=="train":
        short_term_data, long_term_data, news2idx, category2idx = run_preprocessing_pipeline()
        torch.save({
            "short_term_data": short_term_data,
            "long_term_data": long_term_data,
            "news2idx": news2idx,
            "category2idx": category2idx
        },  save_path)

    elif pv.MODE=="test":
        short_term_data, long_term_data = run_test_preprocessing_pipeline()
        torch.save({
            "short_term_data": short_term_data,
            "long_term_data": long_term_data,
        }, save_path)
    
    print("\n========== PREPROCESSING DONE ==========\n")
