import torch
import torch.nn as nn

from models.ltc_encoder import LTCEncoder
from models.embeddings import JointEmbedding
from preprocessing.short_term_preprocessing import run_short_term_pipeline
from preprocessing.utils import N, alpha


class ShortTermEmbedding(nn.Module):
    """
    Short-term preference modeling (without LTC).

    Responsibilities:
    - Embedding lookup for recent interactions
    - Apply hybrid mask
    - Build masked interaction sequence X
    - Return (X, delta_t) for LTC
    """

    def __init__(
        self,
        num_news: int,
        num_categories: int,
        news_dim: int = 64,
        category_dim: int = 16
    ):
        super().__init__()

        self.embedding_layer = JointEmbedding(
            num_news=num_news,
            num_categories=num_categories,
            news_dim=news_dim,
            category_dim=category_dim
        )

        self.output_dim = news_dim + category_dim

    def forward(self, short_term_sequence):
        """
        Args:
            short_term_sequence:
                [
                  (news_idx, category_idx, delta_t, mask),
                  ...
                ]

        Returns:
            X: Tensor of shape (N, D)   -> masked interaction embeddings
            delta_t: Tensor of shape (N,) -> time gaps between interactions
        """

        news_ids = torch.tensor(
            [x[0] for x in short_term_sequence],
            dtype=torch.long
        ).unsqueeze(0)  # (1, N)

        category_ids = torch.tensor(
            [x[1] for x in short_term_sequence],
            dtype=torch.long
        ).unsqueeze(0)  # (1, N)

        delta_t = torch.tensor(
            [x[2] for x in short_term_sequence],
            dtype=torch.float32
        )  # (N,)

        mask = torch.tensor(
            [x[3] for x in short_term_sequence],
            dtype=torch.float32
        ).unsqueeze(-1)  # (N, 1)

        # Embedding lookup
        emb = self.embedding_layer(news_ids, category_ids)  # (1, N, D)
        emb = emb.squeeze(0)                                # (N, D)

        # Apply hybrid mask
        X = emb * mask                                     # (N, D)

        return X, delta_t


class ShortTermLTC(nn.Module):
    """
    Complete short-term preference pipeline:
    - Embedding extraction
    - LTC encoding
    
    Combines ShortTermModel and LTCEncoder in a single end-to-end module.
    """
    
    def __init__(
        self,
        num_news: int,
        num_categories: int,
        news_dim: int = 64,
        category_dim: int = 16,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.short_term_embedding = ShortTermEmbedding(
            num_news=num_news,
            num_categories=num_categories,
            news_dim=news_dim,
            category_dim=category_dim
        )
        
        embedding_dim = news_dim + category_dim
        self.ltc_encoder = LTCEncoder(embedding_dim, hidden_dim)
    
    def forward(self, short_term_sequence):
        """
        Args:
            short_term_sequence:[(news_idx, category_idx, delta_t, mask), ...]
        
        Returns:
            encoded: LTC encoded user representation of shape (hidden_dim,)
            X: Tensor of shape (N, D) - masked interaction embeddings (for inspection)
            delta_t: Tensor of shape (N,) - time gaps (for inspection)
        """
        X, delta_t = self.short_term_embedding(short_term_sequence)
        encoded = self.ltc_encoder(X, delta_t)
        
        return encoded, X, delta_t
    
def short_term_pipeline(num_news, num_categories, news_dim, category_dim, hidden_dim):
    print("Running short-term preprocessing...")
    short_term_data = run_short_term_pipeline(N, alpha)

    print(f"Total users: {len(short_term_data)}")

    model = ShortTermLTC(
        num_news=num_news,        
        num_categories=num_categories,
        news_dim=news_dim,
        category_dim=category_dim,
        hidden_dim=hidden_dim
    )
    model.eval()
    short_term_user_vectors = {}

    with torch.no_grad():
        for user_id, seq in short_term_data.items():

            if len(seq) < 2:
                continue

            encoded, _, _ = model(seq)
            short_term_user_vectors[user_id] = encoded

    #sample printing
    sample_user = next(iter(short_term_user_vectors))
    print("\nSample user:", sample_user)
    print("Encoded vector shape:", short_term_user_vectors[sample_user].shape)
    print("First 10 values:")
    print(short_term_user_vectors[sample_user][:10])
    
if __name__ == "__main__":
    short_term_pipeline(
        num_news=50000,       
        num_categories=50,
        news_dim=64,
        category_dim=16,
        hidden_dim=64
    )

