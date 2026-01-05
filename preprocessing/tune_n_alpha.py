from build_user_interaction_sequence import build_user_interaction_sequences
from collections import Counter
import matplotlib.pyplot as plt
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
    user_sequences = build_user_interaction_sequences("../data/MINDsmall_train")

    N_values = np.arange(5, 50, 5)
    N_results = analyze_N(user_sequences, N_values)

    N_vals = list(N_results.keys())
    coverage = [N_results[n]["coverage_ratio"] for n in N_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(N_vals, coverage, marker='o')
    plt.xlabel('N (number of recent interactions)')
    plt.ylabel('Coverage Ratio')
    plt.title('User Coverage by Minimum Interactions Required')
    plt.grid(True)
    plt.show()

    alpha_values = np.arange(0.1, 0.80, 0.02)
    #print(analyze_alpha(user_interactions, N=10, alpha_values=alpha_values))
 
    alpha_results = analyze_alpha(user_sequences, N=10, alpha_values=alpha_values)
    
    alphas = list(alpha_results.keys())
    avg_dominant = [alpha_results[a]["avg_dominant_categories"] for a in alphas]
    zero_dominant = [alpha_results[a]["zero_dominant_ratio"] for a in alphas]
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, avg_dominant, marker='o', linewidth=2, label='Avg Dominant Categories')
    plt.plot(alphas, zero_dominant, marker='s', linewidth=2, label='Zero Dominant Ratio')
    plt.xlabel('Alpha (threshold ratio)')
    plt.ylabel('Value')
    plt.title('Alpha Analysis: Average Dominant Categories vs Zero Dominant Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()
