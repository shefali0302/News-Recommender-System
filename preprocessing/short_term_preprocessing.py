from preprocessing.utils import get_last_n_interactions


def run_short_term_preprocessing(N, alpha, user_interactions_with_dt):

    print("\n========== SHORT_TERM PREPROCESSING START ==========\n")
    
    # Get Last N Interactions (Sliding Window)    
    print(f"Extracting last N={N} interactions...")
    user_recent_interactions = get_last_n_interactions(user_interactions_with_dt, N)

    print("\nDEBUG: After last N extraction")
    print("Total users:", len(user_recent_interactions))

    sample_user = next(iter(user_recent_interactions))
    print("Sample user:", sample_user)
    print("Interactions:", user_recent_interactions[sample_user])


    print("========== SHORT-TERM PREPROCESSING END ==========\n")

    return user_recent_interactions


