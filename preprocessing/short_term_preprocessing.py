import numpy as np
from .dataset_ingestion import get_user_interactions_with_dt
from .dataset_ingestion import extract_sliding_window
from .dataset_ingestion import detect_dominant_categories


def compute_time_thresholds(user_interactions_with_dt):
    """
    Computes 50th and 75th percentile of Δt values (in seconds)
    Returns both thresholds
    """
    all_dts = []

    for interactions in user_interactions_with_dt.values():
        for _, _, _, dt in interactions:
            if dt > 0:
                all_dts.append(dt)

    all_dts = np.array(all_dts)

    tau_50 = np.percentile(all_dts, 50)
    tau_75 = np.percentile(all_dts, 75)

    return tau_50, tau_75


def compute_short_term_retention(user_recent_interactions, tau):
    """
    Computes Short-Term Retention Ratio (STRR) for a given τ
    """
    total = 0
    retained = 0

    for interactions in user_recent_interactions.values():
        for _, _, _, delta_t in interactions:
            total += 1
            if delta_t <= tau:
                retained += 1

    return retained / total if total > 0 else 0


def select_best_threshold(user_recent_interactions, tau_50, tau_75):
    """
    Chooses the better τ based on short-term retention metric
    """
    strr_50 = compute_short_term_retention(user_recent_interactions, tau_50)
    strr_75 = compute_short_term_retention(user_recent_interactions, tau_75)

    print(f"STRR @ τ50 ({tau_50:.2f}s): {strr_50:.4f}")
    print(f"STRR @ τ75 ({tau_75:.2f}s): {strr_75:.4f}")

    # Prefer higher retention but avoid extreme looseness
    if strr_75 - strr_50 >= 0.05:
        chosen_tau = tau_75
        reason = "Higher short-term retention with acceptable recency"
    else:
        chosen_tau = tau_50
        reason = "More conservative threshold with sufficient retention"

    print(f"Chosen τ: {chosen_tau:.2f}s → {reason}")

    return chosen_tau


def apply_time_mask(user_recent_interactions, tau):
    """
    Applies time-based masking
    """
    user_time_masks = {}

    for user_id, interactions in user_recent_interactions.items():
        masked = []
        for news_id, timestamp, category, delta_t in interactions:
            m_time = 1 if delta_t <= tau else 0
            masked.append((news_id, category, delta_t, m_time))

        user_time_masks[user_id] = masked

    return user_time_masks

def apply_category_mask(user_time_masked, user_dominant_categories):
    """
    Input:
      user_time_masked:
        dict {user_id: [(news_id, category, delta_t, m_time), ...]}

      user_dominant_categories:
        dict {user_id: set(categories)}

    Output:
      dict {user_id: [(news_id, category, delta_t, m_time, m_cat), ...]}
    """

    user_category_masked = {}

    for user_id, interactions in user_time_masked.items():
        dominant_categories = user_dominant_categories[user_id]
        masked = []

        for news_id, category, delta_t, m_time in interactions:
            m_cat = 1 if category in dominant_categories else 0
            masked.append(
                (news_id, category, delta_t, m_time, m_cat)
            )

        user_category_masked[user_id] = masked

    return user_category_masked

def apply_hybrid_mask(user_category_masked):
    """
    Output:
      dict {user_id: [(news_id, category, delta_t, m), ...]}
    """

    user_hybrid_masked = {}

    for user_id, interactions in user_category_masked.items():
        hybrid = []

        for news_id, category, delta_t, m_time, m_cat in interactions:
            m = m_time * m_cat
            hybrid.append(
                (news_id, category, delta_t, m)
            )

        user_hybrid_masked[user_id] = hybrid

    return user_hybrid_masked

def initialize_embeddings(news_ids, categories, news_dim=64, category_dim=16):
    """
    Returns:
      news_embedding_matrix: dict {news_id: vector}
      category_embedding_matrix: dict {category: vector}
    """

    news_embedding_matrix = {
        nid: np.random.randn(news_dim)
        for nid in news_ids
    }

    category_embedding_matrix = {
        cat: np.random.randn(category_dim)
        for cat in categories
    }

    return news_embedding_matrix, category_embedding_matrix

def build_dense_interaction_vectors(
    user_hybrid_masked,
    news_embedding_matrix,
    category_embedding_matrix
):
    """
    Output:
      dict {user_id: [dense_vector, ...]}
    """

    user_dense_vectors = {}

    for user_id, interactions in user_hybrid_masked.items():
        vectors = []

        for news_id, category, delta_t, m in interactions:
            news_vec = news_embedding_matrix[news_id]
            cat_vec = category_embedding_matrix[category]

            combined = np.concatenate([news_vec, cat_vec])

            # apply hybrid mask
            combined = m * combined

            vectors.append(combined)

        user_dense_vectors[user_id] = vectors

    return user_dense_vectors


if __name__ == "__main__":
    print("Loading user interactions with time gaps...")
    user_interactions_with_dt = get_user_interactions_with_dt()

    print("Extracting sliding window...")
    N = 10
    user_recent_interactions = extract_sliding_window(
        user_interactions_with_dt, N
    )

    print("Detecting dominant categories...")
    alpha = 0.4
    user_dominant_categories = detect_dominant_categories(
        user_recent_interactions, N, alpha
    )

    print("Computing percentile-based thresholds...")
    tau_50, tau_75 = compute_time_thresholds(user_interactions_with_dt)

    print("Selecting best threshold using short-term retention metric...")
    tau = select_best_threshold(
        user_recent_interactions, tau_50, tau_75
    )

    print("Applying time-based masking...")
    user_time_masked = apply_time_mask(
        user_recent_interactions, tau
    )

    print("Applying category-based masking...")
    user_category_masked = apply_category_mask(
        user_time_masked,
        user_dominant_categories
    )

    print("Applying hybrid masking...")
    user_hybrid_masked = apply_hybrid_mask(user_category_masked)

    print("Initializing embeddings...")
    all_news_ids = set()
    all_categories = set()

    for interactions in user_hybrid_masked.values():
        for news_id, category, _, _ in interactions:
            all_news_ids.add(news_id)
            all_categories.add(category)

    news_emb, category_emb = initialize_embeddings(
        all_news_ids, all_categories
    )

    print("Building dense interaction vectors...")
    user_dense_vectors = build_dense_interaction_vectors(
        user_hybrid_masked,
        news_emb,
        category_emb
    )

    # ONE sanity check only
    sample_user = next(iter(user_dense_vectors))
    print("\nSample user:", sample_user)
    print("Dense vector shape:", user_dense_vectors[sample_user][0].shape)
