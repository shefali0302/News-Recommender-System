import torch
from models.short_term import ShortTermPipeline
from preprocessing.short_term_preprocessing import run_short_term_pipeline
from preprocessing.utils import N, alpha

# -----------------------------
# Run short-term preprocessing
# -----------------------------
print("Running short-term preprocessing...")
short_term_data = run_short_term_pipeline(N, alpha)

print(f"Total users: {len(short_term_data)}")

# -----------------------------
# Initialize model ONCE
# -----------------------------
model = ShortTermPipeline(
    num_news=50000,        # safe upper bound
    num_categories=50,
    news_dim=64,
    category_dim=16,
    hidden_dim=64
)

# model.eval()  # important: inference mode

# # -----------------------------
# # Run LTC encoding for ALL users
# # -----------------------------
# short_term_user_vectors = {}

# with torch.no_grad():
#     for user_id, sequence in short_term_data.items():

#         if len(sequence) == 0:
#             continue

#         encoded, X, delta_t = model(sequence)

#         # encoded shape: (hidden_dim,)
#         short_term_user_vectors[user_id] = encoded

model.eval()
short_term_user_vectors = {}

with torch.no_grad():
    for user_id, seq in short_term_data.items():

        if len(seq) < 2:
            continue

        encoded, _, _ = model(seq)
        short_term_user_vectors[user_id] = encoded


# -----------------------------
# Sanity check
# -----------------------------
sample_user = next(iter(short_term_user_vectors))

print("\nSample user:", sample_user)
print("Encoded vector shape:",
      short_term_user_vectors[sample_user].shape)

print("First 10 values:")
print(short_term_user_vectors[sample_user][:10])
