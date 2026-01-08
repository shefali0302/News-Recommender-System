import numpy as np
from datetime import datetime

from models.long_term import (
    chunk_interactions_by_day,
    mean_pool_daily_chunks,
    build_daily_sequence_with_time_gaps
)

def test_long_term_pipeline():
    # Fake interaction embeddings
    user_interaction_embeddings = {
        "U1": [
            (datetime(2023, 10, 1, 10, 0), np.random.rand(128)),
            (datetime(2023, 10, 1, 15, 0), np.random.rand(128)),
            (datetime(2023, 10, 2, 9, 0),  np.random.rand(128)),
        ]
    }

    daily_chunks = chunk_interactions_by_day(user_interaction_embeddings)
    assert len(daily_chunks["U1"]) == 2

    daily_vectors = mean_pool_daily_chunks(daily_chunks)
    for vec in daily_vectors["U1"].values():
        assert vec.shape == (128,)

    daily_sequence = build_daily_sequence_with_time_gaps(daily_vectors)
    assert daily_sequence["U1"][0][1] == 0
    assert daily_sequence["U1"][1][1] == 1

    print("✅ Long-term pipeline sanity test passed!")


if __name__ == "__main__":
    test_long_term_pipeline()
