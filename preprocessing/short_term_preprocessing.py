"""
Short-term preprocessing pipeline
ORCHESTRATION ONLY
"""

import os

from preprocessing.dataset_ingestion import load_news_categories, load_user_interactions, NEWS_PATH
from preprocessing.dataset_ingestion import build_id_mappings, map_interactions_to_indices
from preprocessing.sequence_builder import sort_user_interactions, compute_time_gaps
from preprocessing.utils import get_last_n_interactions, compute_dominant_categories,   compute_time_thresholds, apply_time_mask, apply_category_mask, apply_hybrid_mask, N, alpha


def run_short_term_pipeline(N, alpha):
    print("\n========== SHORT-TERM PREPROCESSING START ==========\n")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # --------------------------------------------------
    # 1. Load News Categories
    # --------------------------------------------------
    print("Loading news categories...")
    news_category_map = load_news_categories(NEWS_PATH)

    # --------------------------------------------------
    # 2. Load Raw User Interactions 
    # --------------------------------------------------
    print("Loading raw user interactions...")
    behaviors_path = os.path.join(BASE_DIR, "..", "data", "MINDsmall_train", "behaviors.tsv")
    
    user_interactions = load_user_interactions(behaviors_path, news_category_map)
    news2idx, cat2idx = build_id_mappings(user_interactions)
    user_interactions = map_interactions_to_indices(user_interactions, news2idx, cat2idx)
    
    # --------------------------------------------------
    # 3. Sort Interactions by Time
    # --------------------------------------------------
    print("Sorting interactions by timestamp...")
    user_interactions = sort_user_interactions(user_interactions)

    # --------------------------------------------------
    # 4. Compute Time Gaps (Δt)
    # --------------------------------------------------
    print("Computing time gaps (Δt)...")
    user_interactions_with_dt = compute_time_gaps(user_interactions)

    # --------------------------------------------------
    # 5. Get Last N Interactions (Sliding Window)
    # --------------------------------------------------
    print(f"Extracting last N={N} interactions...")
    user_recent_interactions = get_last_n_interactions(user_interactions_with_dt, N)

    # --------------------------------------------------
    # 6. Detect Dominant Categories
    # --------------------------------------------------
    print(f"Detecting dominant categories (alpha={alpha})...")
    user_dominant_categories = {}
    for user_id, interactions in user_recent_interactions.items():
        user_dominant_categories[user_id] = set(
            compute_dominant_categories(interactions, alpha)
        )

    # --------------------------------------------------
    # 7. Compute Time Thresholds (τ)
    # --------------------------------------------------
    print("Computing percentile-based time thresholds...")
    tau_50, tau_75 = compute_time_thresholds(user_interactions_with_dt)

    print(f"τ50 (median Δt): {tau_50:.2f} seconds")
    print(f"τ75 (75th percentile Δt): {tau_75:.2f} seconds")

    tau = tau_50  # conservative default
    print(f"Selected τ = {tau:.2f} seconds\n")

    # --------------------------------------------------
    # 8. Apply Time Mask
    # --------------------------------------------------
    print("Applying time-based mask...")
    user_time_masked = apply_time_mask(user_recent_interactions, tau)

    # --------------------------------------------------
    # 9. Apply Category Mask
    # --------------------------------------------------
    print("Applying category-based mask...")
    user_category_masked = apply_category_mask(user_time_masked, user_dominant_categories)

    # --------------------------------------------------
    # 10. Apply Hybrid Mask
    # --------------------------------------------------
    print("Applying hybrid (time × category) mask...")
    user_hybrid_masked = apply_hybrid_mask(user_category_masked)

    print("========== SHORT-TERM PREPROCESSING END ==========\n")

    return user_hybrid_masked


if __name__ == "__main__":
    run_short_term_pipeline(N, alpha) 
