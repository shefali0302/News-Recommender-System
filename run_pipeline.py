# ----------------------------------------------------------------------
# This is just a test script to run the entire pipeline and 
# check if all components are working together correctly.
# THIS IS ONLY FOR TESTING PURPOSES AND NOT THE FINAL TRAINING SCRIPT.
# -----------------------------------------------------------------------

from preprocessing.run_preprocessing_pipeline import run_preprocessing_pipeline
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC
from models.fusion import FusionGate
from models.scoring import ItemScorer, ItemScorer
import torch
import torch.nn.functional as F


def main():
    short_term_data, long_term_data, news2idx, category2idx = run_preprocessing_pipeline()

    num_news = max(news2idx.values()) + 1
    num_categories = max(category2idx.values()) + 1

    print("Num news:", num_news)
    print("Num categories:", num_categories)

    short_model = ShortTermLTC(
        num_news=num_news,
        num_categories=num_categories,
        hidden_dim=64
    )

    long_model = LongTermLTC(
        num_news=num_news,
        num_categories=num_categories,
        hidden_dim=64
    )

    user_id = next(iter(short_term_data))

    st_vec, _, _ = short_model(short_term_data[user_id])
    lt_vec, _, _ = long_model(long_term_data[user_id])

    print("ST vector shape:", st_vec.shape)
    print("LT vector shape:", lt_vec.shape)

    fusion_gate = FusionGate(dim=64)

    fused_user_vec, gate_value = fusion_gate(st_vec, lt_vec)

    print("Fused user vector shape:", fused_user_vec.shape)
    print("Gate value:", gate_value.item())

    candidate_news_ids = [10, 20, 35, 50, 100]   # sample impression list
    clicked_index = 2   # suppose user clicked item index 2

    scorer = ItemScorer(num_news=num_news, embedding_dim=64)

    scores, probs = scorer(fused_user_vec, candidate_news_ids)

    print("Scores:", scores)
    print("Probabilities:", probs)

    target = torch.tensor(clicked_index)
    loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))

    print("Loss:", loss.item())


if __name__ == "__main__":
    main()
