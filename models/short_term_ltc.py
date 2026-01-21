import torch
import torch.nn as nn
from ncps.torch import LTC
from models.embeddings import build_short_term_embeddings
import torch
import numpy as np

class ShortTermLTCEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.ltc = LTC(
            input_size=embedding_dim,
            units=hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, N, embedding_dim)
        """
        outputs, final_hidden = self.ltc(x)
        return final_hidden

#it is taking each individual embedding vector and encoding it using LTC
def encode_user_interactions(user_dense_vectors, hidden_dim=64, batch_size=32):
    """
    Convert embeddings to LTC encoded representations
    
    Args:
        user_dense_vectors: dict {user_id: [dense_vector, ...]}
        hidden_dim: hidden dimension (default 64, reduced for memory efficiency)
        batch_size: batch size for processing (default 32)
    
    Returns:
        dict {user_id: encoded_tensor}
    """
    embedding_dim = next(iter(user_dense_vectors.values()))[0].shape[0]
    encoder = ShortTermLTCEncoder(embedding_dim, hidden_dim)
    encoder.eval() #eval mode reduces memory usage
    
    encoded = {}
    user_ids = list(user_dense_vectors.keys())

    #batch processing for memory efficiency
    with torch.no_grad(): 
        for i in range(0, len(user_ids), batch_size):
            batch_user_ids = user_ids[i:i+batch_size]
            for user_id in batch_user_ids:
                vectors = user_dense_vectors[user_id]
        
                x = torch.tensor(np.array(vectors), dtype=torch.float32).unsqueeze(0)
                encoded[user_id] = encoder(x).squeeze(0)
    
    return encoded


if __name__ == "__main__":
    user_dense_vectors = build_short_term_embeddings()
    encoded_users = encode_user_interactions(user_dense_vectors)
    
    # # Print results in a readable format
    # print("Encoded User Interactions:")
    # print("-" * 50)
    # for user_id, encoded_tensor in encoded_users.items():
    #     print(f"User ID: {user_id}")
    #     print(f"  Encoded Shape: {encoded_tensor.shape}")
    #     print(f"  Encoded Values: {encoded_tensor.detach().numpy()}")
    #     print()

    
