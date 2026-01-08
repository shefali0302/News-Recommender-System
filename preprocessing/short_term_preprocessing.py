"""
Short-term preprocessing pipeline
ORCHESTRATION ONLY – zero redundant logic
"""

import os

from dataset_ingestion import load_news_categories, load_user_interactions, NEWS_PATH
from sequence_builder import sort_user_interactions, compute_time_gaps
from utils import (
    get_last_n_interactions,
    compute_dominant_categories,
    compute_time_thresholds,
    apply_time_mask,
    apply_category_mask,
    apply_hybrid_mask,
    compute_dominant_categories_all_users,
    N,
    alpha
)


def run_short_term_pipeline(N, alpha):
    print("\n========== SHORT-TERM PREPROCESSING START ==========\n")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # --------------------------------------------------
    # 1. Load News Categories
    # --------------------------------------------------
    print("1️⃣ Loading news categories...")
    news_category_map = load_news_categories(NEWS_PATH)
    print(f"✔ Total news articles: {len(news_category_map)}\n")

    # --------------------------------------------------
    # 2. Load Raw User Interactions
    # --------------------------------------------------
    print("2️⃣ Loading raw user interactions...")
    behaviors_path = os.path.join(
        BASE_DIR, "..", "data", "MINDsmall_train", "behaviors.tsv"
    )
    
    user_interactions = load_user_interactions(
        behaviors_path, news_category_map
    )
    

    print(f"✔ Total users: {len(user_interactions)}")
    sample_user = next(iter(user_interactions))
    print(f"🔍 Sample user (raw): {sample_user}")
    print(user_interactions[sample_user][:5], "\n")
    
    # --------------------------------------------------
    # 3. Sort Interactions by Time
    # --------------------------------------------------
    print("3️⃣ Sorting interactions by timestamp...")
    user_interactions = sort_user_interactions(user_interactions)

    print(f"🔍 Sample user (sorted): {sample_user}")
    print(user_interactions[sample_user][:5], "\n")

    # --------------------------------------------------
    # 4. Compute Time Gaps (Δt)
    # --------------------------------------------------
    print("4️⃣ Computing time gaps (Δt)...")
    user_interactions_with_dt = compute_time_gaps(user_interactions)

    print(f"🔍 Sample user (with Δt): {sample_user}")
    print(user_interactions_with_dt[sample_user][:5], "\n")

    # --------------------------------------------------
    # 5. Get Last N Interactions (Sliding Window)
    # --------------------------------------------------
    print(f"5️⃣ Extracting last N={N} interactions...")
    user_recent_interactions = get_last_n_interactions(
        user_interactions_with_dt, N
    )

    print(f"🔍 Sample user (last {N}): {sample_user}")
    print(user_recent_interactions[sample_user], "\n")

    # --------------------------------------------------
    # 6. Detect Dominant Categories
    # --------------------------------------------------
    print(f"6️⃣ Detecting dominant categories (alpha={alpha})...")
    user_dominant_categories = compute_dominant_categories_all_users(
        user_recent_interactions, alpha
    )

    print(f"🔍 Sample user dominant categories:")
    print(user_dominant_categories[sample_user], "\n")

    # --------------------------------------------------
    # 7. Compute Time Thresholds (τ)
    # --------------------------------------------------
    print("7️⃣ Computing percentile-based time thresholds...")
    tau_50, tau_75 = compute_time_thresholds(user_interactions_with_dt)

    print(f"τ50 (median Δt): {tau_50:.2f} seconds")
    print(f"τ75 (75th percentile Δt): {tau_75:.2f} seconds")

    tau = tau_50  # conservative default
    print(f"✔ Selected τ = {tau:.2f} seconds\n")

    # --------------------------------------------------
    # 8. Apply Time Mask
    # --------------------------------------------------
    print("8️⃣ Applying time-based mask...")
    user_time_masked = apply_time_mask(
        user_recent_interactions, tau
    )

    print(f"🔍 Sample user (time-masked):")
    print(user_time_masked[sample_user], "\n")

    # --------------------------------------------------
    # 9. Apply Category Mask
    # --------------------------------------------------
    print("9️⃣ Applying category-based mask...")
    user_category_masked = apply_category_mask(
        user_time_masked, user_dominant_categories
    )

    print(f"🔍 Sample user (category-masked):")
    print(user_category_masked[sample_user], "\n")

    # --------------------------------------------------
    # 10. Apply Hybrid Mask
    # --------------------------------------------------
    print("🔟 Applying hybrid (time × category) mask...")
    user_hybrid_masked = apply_hybrid_mask(user_category_masked)

    print(f"🔍 Sample user (hybrid-masked):")
    print(user_hybrid_masked[sample_user], "\n")

    print("========== SHORT-TERM PREPROCESSING END ==========\n")

    return user_hybrid_masked


if __name__ == "__main__":
    run_short_term_pipeline(N=10, alpha=0.4) 
