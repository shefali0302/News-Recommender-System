
def compute_time_gaps(user_interactions):
    """
    Input:
      dict {user_id: [(news_id, timestamp, category), ...]}

    Output:
      dict {user_id: [(news_id, timestamp, category, delta_t), ...]}
      where delta_t is time gap in seconds
    """

    user_interactions_with_dt = {}

    for user_id, interactions in user_interactions.items():
        enriched_interactions = []

        prev_time = None

        for news_id, timestamp, category in interactions:
            if prev_time is None:
                delta_t = 0.0  # first interaction
            else:
                delta_t = (timestamp - prev_time).total_seconds()

            enriched_interactions.append(
                (news_id, timestamp, category, delta_t)
            )

            prev_time = timestamp

        user_interactions_with_dt[user_id] = enriched_interactions

    return user_interactions_with_dt