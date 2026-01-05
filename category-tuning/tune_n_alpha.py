from preprocessing.dataset_ingestion import build_user_interactions
from collections import Counter
import numpy as np
import math



def analyze_N(user_interactions, N_values):
    results = {}

    total_users = len(user_interactions)

    for N in N_values:
        users_with_enough = sum(
            1 for interactions in user_interactions.values()
            if len(interactions) >= N
        )

        coverage = users_with_enough / total_users

        results[N] = {
            "users_with_>=N": users_with_enough,
            "coverage_ratio": round(coverage, 3)
        }

    return results


def analyze_alpha(user_interactions, N, alpha_values):
    alpha_results = {}

    for alpha in alpha_values:
        dominant_counts = []

        for interactions in user_interactions.values():
            if len(interactions) < N:
                continue

            recent = interactions[-N:]
            categories = [x[2] for x in recent]

            freq = Counter(categories)
            theta = math.ceil(alpha * N)

            dominant = [
                cat for cat, count in freq.items()
                if count >= theta
            ]

            dominant_counts.append(len(dominant))

        alpha_results[alpha] = {
            "avg_dominant_categories": round(
                np.mean(dominant_counts), 2
            ),
            "zero_dominant_ratio": round(
                sum(1 for x in dominant_counts if x == 0) / len(dominant_counts),
                3
            )
        }

    return alpha_results
   
if __name__ == "__main__":
    user_interactions = build_user_interactions()

    N_values = [5, 10, 15, 20, 25, 30]
    print(analyze_N(user_interactions, N_values))

    alpha_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7]
    print(analyze_alpha(user_interactions, N=10, alpha_values=alpha_values))
