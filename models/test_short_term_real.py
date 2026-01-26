from models.short_term import ShortTermModel
from preprocessing.short_term_preprocessing import run_short_term_pipeline
from preprocessing.utils import N, alpha

# -----------------------------
# Run short-term preprocessing
# -----------------------------
print("Running short-term preprocessing...")
short_term_data = run_short_term_pipeline(N, alpha)

# Pick one user
sample_user = next(iter(short_term_data))
short_term_sequence = short_term_data[sample_user]

print("Sample user:", sample_user)
print("Number of recent interactions:", len(short_term_sequence))

print("\nFirst 3 interactions (news_idx, cat_idx, Δt, mask):")
for x in short_term_sequence[:3]:
    print(x)

# -----------------------------
# Short-term model
# -----------------------------
model = ShortTermModel(
    num_news=50000,        # safe upper bound
    num_categories=50
)

X, delta_t = model(short_term_sequence)

print("\nX shape:", X.shape)
print("delta_t shape:", delta_t.shape)

print("\nFirst interaction embedding (first 10 values):")
print(X[0][:10])
