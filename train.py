import torch
import torch.optim as optim

from preprocessing.run_preprocessing_pipeline import run_preprocessing_pipeline
from models.short_term import ShortTermLTC
from models.long_term import LongTermLTC
from models.fusion import FusionGate
from models.scoring import ItemScorer
from training.loss import compute_loss
from inference.top_k_recs import get_top_k_news


MAX_USERS=20

def train_model(num_epochs):
    short_term_data, long_term_data, news2idx, category2idx = run_preprocessing_pipeline()

    num_news = max(news2idx.values()) + 1
    num_categories = max(category2idx.values()) + 1

    short_model = ShortTermLTC(num_news, num_categories, hidden_dim=64)
    long_model  = LongTermLTC(num_news, num_categories, hidden_dim=64)
    fusion_gate = FusionGate(dim=64)
    scorer      = ItemScorer(num_news=num_news, embedding_dim=64)

    optimizer = optim.Adam(
        list(short_model.parameters()) +
        list(long_model.parameters()) +
        list(fusion_gate.parameters()) +
        list(scorer.parameters()),
        lr=0.001
    )

    for epoch in range(num_epochs):

        total_loss = 0

        for i, user_id in enumerate(short_term_data):

            if i >= MAX_USERS:
                break

            if user_id not in long_term_data:
                continue

            short_seq = short_term_data[user_id]
            long_seq  = long_term_data[user_id]

            if len(short_seq) < 2:
                continue

            # candidates = news in short-term window
            candidates = [x[0] for x in short_seq]

            # target = last clicked item
            clicked_index = len(candidates) - 1

            st_vec, _, _ = short_model(short_seq)
            lt_vec, _, _ = long_model(long_seq)

            fused_vec, _ = fusion_gate(st_vec, lt_vec)

            scores, probs = scorer(fused_vec, candidates)

            loss = compute_loss(scores, clicked_index)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss = {total_loss:.4f}")

    print("Training complete")

    user_id = next(iter(short_term_data))

    st_vec, _, _ = short_model(short_term_data[user_id])
    lt_vec, _, _ = long_model(long_term_data[user_id])
    fused_vec, _ = fusion_gate(st_vec, lt_vec)

    candidates = [x[0] for x in short_term_data[user_id]]
    scores, probs = scorer(fused_vec, candidates)

    top_k = get_top_k_news(scores, candidates, k=5)

    print("Top K:", top_k)

    return short_model, long_model, fusion_gate, scorer


if __name__ == "__main__":
    train_model(num_epochs=3)

    
