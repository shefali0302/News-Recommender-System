import torch
import random
import path_variables as pv
from sklearn.model_selection import train_test_split

def split_preprocessed_pt(
    input_path=pv.MINDS_PREPROCESSED_TRAIN,
    train_out=pv.MINDS_PREPROCESSED_TRAIN_TRAIN,
    val_out=pv.MINDS_PREPROCESSED_TRAIN_VAL,
    val_ratio=0.2,
    seed=42
):

    # Load full dataset
    data = torch.load(input_path)

    short_term_data = data["short_term_data"]
    long_term_data = data["long_term_data"]
    news2idx = data["news2idx"]
    category2idx = data["category2idx"]

    # Get all users
    all_users = list(short_term_data.keys())

    # User-level split
    train_users, val_users = train_test_split(
        all_users,
        test_size=val_ratio,
        random_state=seed
    )

    # Create split dictionaries
    train_short = {u: short_term_data[u] for u in train_users}
    val_short = {u: short_term_data[u] for u in val_users}

    train_long = {u: long_term_data[u] for u in train_users}
    val_long = {u: long_term_data[u] for u in val_users}

    # Save train
    torch.save({
        "short_term_data": train_short,
        "long_term_data": train_long,
        "news2idx": news2idx,
        "category2idx": category2idx
    }, train_out)

    # Save validation
    torch.save({
        "short_term_data": val_short,
        "long_term_data": val_long,
        "news2idx": news2idx,
        "category2idx": category2idx
    }, val_out)

    print("Split complete.")
    print("Train users:", len(train_users))
    print("Val users:", len(val_users))


# Run it
if __name__ == "__main__":
    split_preprocessed_pt()
