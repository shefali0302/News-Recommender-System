import torch
import torch.nn as nn
from ncps.torch import LTC


class LTCEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.ltc = LTC(
            input_size=embedding_dim,
            units=hidden_dim,
            batch_first=True
        )

    def forward(self, x, delta_t):
        """
        Args:
            x: Tensor (N, D) or (1, N, D)
            delta_t: Tensor (N,)
        Returns:
            encoded: Tensor (hidden_dim,)
        """

        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)          # (1, N, D)

        # LTC expects timespans between steps
        # Clamp to avoid numerical instability
        timespans = delta_t.clamp(min=1e-3).unsqueeze(0)  # (1, N)

        outputs, final_hidden = self.ltc(
            x,
            timespans=timespans
        )

        # final_hidden shape: (1, hidden_dim)
        return final_hidden.squeeze(0)


# import torch
# import torch.nn as nn
# from ncps.torch import LTC
# import torch
# import numpy as np

# class LTCEncoder(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim):
#         super().__init__()

#         self.ltc = LTC(
#             input_size=embedding_dim,
#             units=hidden_dim,
#             batch_first=True
#         )

#     def forward(self, x, delta_t=None):
#         """
#         Args:
#             x: Tensor of shape (batch_size, N, embedding_dim) or (N, embedding_dim)
#             delta_t: Tensor of shape (N,) - time gaps between interactions (optional)
#         """
#         # If x is 2D (N, D), add batch dimension
#         if x.dim() == 2:
#             x = x.unsqueeze(0)  # (1, N, D)
        
#         outputs, final_hidden = self.ltc(x)
#         return final_hidden

# #take out
# def encode_user_interactions_from_short_term(x, delta_t, hidden_dim=64):
#     """
#     This function directly accepts the output from ShortTermModel.forward()
    
#     Args:
#         x: Tensor of shape (N, D) - masked interaction embeddings from ShortTermModel
#         delta_t: Tensor of shape (N,) - time gaps between interactions
#         hidden_dim: hidden dimension for LTC encoder
    
#     Returns:
#         Tensor of shape (hidden_dim,) - LTC encoded user representation
#     """
#     embedding_dim = x.shape[-1]
#     encoder = LTCEncoder(embedding_dim, hidden_dim)
#     encoder.eval()
    
#     with torch.no_grad():
#         encoded = encoder(x, delta_t)
    
#     return encoded


# # def encode_user_interactions(user_dense_vectors, hidden_dim=64, batch_size=32):
# #     """
# #     Convert embeddings to LTC encoded representations
    
# #     Args:
# #         user_dense_vectors: dict {user_id: [dense_vector, ...]}
# #         hidden_dim: hidden dimension (default 64, reduced for memory efficiency)
# #         batch_size: batch size for processing (default 32)
    
# #     Returns:
# #         dict {user_id: encoded_tensor}
# #     """
# #     embedding_dim = next(iter(user_dense_vectors.values()))[0].shape[0]
# #     encoder = LTCEncoder(embedding_dim, hidden_dim)
# #     encoder.eval() #eval mode reduces memory usage
    
# #     encoded = {}
# #     user_ids = list(user_dense_vectors.keys())

# #     #batch processing for memory efficiency
# #     with torch.no_grad(): 
# #         for i in range(0, len(user_ids), batch_size):
# #             batch_user_ids = user_ids[i:i+batch_size]
# #             for user_id in batch_user_ids:
# #                 vectors = user_dense_vectors[user_id]
        
# #                 x = torch.tensor(np.array(vectors), dtype=torch.float32).unsqueeze(0)
# #                 encoded[user_id] = encoder(x).squeeze(0)
    
# #     return encoded


# # if __name__ == "__main__":
# #     user_dense_vectors = build_short_term_embeddings()
# #     encoded_users = encode_user_interactions(user_dense_vectors)
    
#     # # Print results in a readable format
#     # print("Encoded User Interactions:")
#     # print("-" * 50)
#     # for user_id, encoded_tensor in encoded_users.items():
#     #     print(f"User ID: {user_id}")
#     #     print(f"  Encoded Shape: {encoded_tensor.shape}")
#     #     print(f"  Encoded Values: {encoded_tensor.detach().numpy()}")
#     #     print()

    
