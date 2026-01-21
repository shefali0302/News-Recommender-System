"""
Embedding computation module
Builds dense vector representations from short-term preprocessing output
"""

import numpy as np

from preprocessing.short_term_preprocessing import run_short_term_pipeline
from preprocessing.utils import N, alpha

# Step 1: Extract Vocabulary from Preprocessed Output
def extract_embedding_vocab(user_hybrid_masked):
    """
    Input:
      dict {user_id: [(news_id, category, delta_t, mask), ...]}

    Output:
      news_ids: set
      categories: set
    """
    news_ids = set()
    categories = set()

    for interactions in user_hybrid_masked.values():
        for news_id, category, _, _ in interactions:
            news_ids.add(news_id)
            categories.add(category)

    return news_ids, categories


# Step 2: Initialize Embedding Matrices
def initialize_embeddings(
    news_ids,
    categories,
    news_dim=64,
    category_dim=16,
    seed=42
):
    """
    Returns:
      news_embedding_matrix: dict {news_id: np.ndarray}
      category_embedding_matrix: dict {category: np.ndarray}
    """
    rng = np.random.default_rng(seed)

    news_embedding_matrix = {
        nid: rng.standard_normal(news_dim)
        for nid in news_ids
    }

    category_embedding_matrix = {
        cat: rng.standard_normal(category_dim)
        for cat in categories
    }

    return news_embedding_matrix, category_embedding_matrix


# Step 3: Build Dense Interaction Embeddings

def build_dense_interaction_vectors(
    user_hybrid_masked,
    news_embedding_matrix,
    category_embedding_matrix
):
    """
    Input:
      user_hybrid_masked:
        dict {user_id: [(news_id, category, delta_t, m), ...]}

    Output:
      dict {user_id: [np.ndarray, ...]}
    """
    user_dense_vectors = {}

    for user_id, interactions in user_hybrid_masked.items():
        vectors = []

        for news_id, category, _, m in interactions:
            news_vec = news_embedding_matrix[news_id]
            cat_vec = category_embedding_matrix[category]

            # concatenate news + category
            dense_vec = np.concatenate([news_vec, cat_vec])

            # apply hybrid mask
            dense_vec = m * dense_vec

            vectors.append(dense_vec)

        user_dense_vectors[user_id] = vectors

    return user_dense_vectors


# Step 4: End-to-End Embedding Pipeline
def build_short_term_embeddings(
    news_dim=64,
    category_dim=16
):
    """
    Runs short-term preprocessing and builds embeddings

    Returns:
      user_dense_vectors: dict {user_id: [dense_vector, ...]}
    """

    # Run preprocessing (REUSED, not duplicated)
    user_hybrid_masked = run_short_term_pipeline(N, alpha)

    # Extract vocab
    news_ids, categories = extract_embedding_vocab(user_hybrid_masked)

    # Initialize embeddings
    news_emb, cat_emb = initialize_embeddings(
        news_ids,
        categories,
        news_dim=news_dim,
        category_dim=category_dim
    )

    # Build dense vectors
    user_dense_vectors = build_dense_interaction_vectors(
        user_hybrid_masked,
        news_emb,
        cat_emb
    )

    return user_dense_vectors


# Debug / Sanity Check
if __name__ == "__main__":
    user_dense_vectors = build_short_term_embeddings()

    sample_user = next(iter(user_dense_vectors))
    sample_vectors = user_dense_vectors[sample_user]

    print("Sample user:", sample_user)
    print("Dense vector shape:", user_dense_vectors[sample_user][0].shape)
    print("Number of interactions:", len(sample_vectors))
    print("Embedding dimension:", sample_vectors[0].shape)
    print("First embedding (first 10 values):")
    print(sample_vectors[0][:10])