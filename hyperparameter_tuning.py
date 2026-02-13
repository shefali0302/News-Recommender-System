import os
import torch
from utils import load_data
from training.experiment_util import create_experiment_folder
from train import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


if __name__ == "__main__":
    MODE = "full"  

    train_data = load_data("data/MINDsmall_train_preprocessed_train.pt")
    dev_data = load_data("data/MINDsmall_train_preprocessed_val.pt")

    train_short = train_data["short_term_data"]
    train_long = train_data["long_term_data"]
    news2idx = train_data["news2idx"]
    category2idx = train_data["category2idx"]

    dev_short = dev_data["short_term_data"]
    dev_long = dev_data["long_term_data"]

    learning_rates = [0.001, 0.0005]
    embedding_dims = [64]
    hidden_dims = [64]
    num_epochs = 10

    best_score = 0
    best_config = None
    best_exp_dir = None

    print(f"Starting hyperparameter tuning with {len(learning_rates) * len(embedding_dims) * len(hidden_dims)} configurations...\n")

    for lr in learning_rates:
        for emb_dim in embedding_dims:
            for hid_dim in hidden_dims:

                config = {
                    "embedding_dim": emb_dim,
                    "hidden_dim": hid_dim,
                    "learning_rate": lr,
                    "num_epochs": num_epochs,
                    "mode": MODE,
                    "optimizer": "Adam",
                    "device": str(device),
                    "dataset": "MINDsmall"
                }

                print(f"Training with config: lr={lr}, emb_dim={emb_dim}, hid_dim={hid_dim}")
                
                exp_dir = create_experiment_folder(config)

                short_model, long_model, fusion_gate, scorer = train_model(
                    train_short, train_long,
                    dev_short, dev_long,
                    news2idx, category2idx,
                    num_epochs=num_epochs,
                    exp_dir=exp_dir,
                    embedding_dim=emb_dim,
                    hidden_dim=hid_dim,
                    learning_rate=lr
                )

                val_metrics = evaluate_model(
                    short_model,
                    long_model,
                    fusion_gate,
                    scorer,
                    dev_short,
                    dev_long,
                    MODE
                )

                score = val_metrics["NDCG@10"]
                print(f"Validation NDCG@10: {score:.4f}")
                print(f"Full metrics: {val_metrics}\n")

                if score > best_score:
                    best_score = score
                    best_config = config
                    best_exp_dir = exp_dir
                    print(f"New best configuration found!\n")

    print("\n" + "="*60)
    print("Hyperparameter Tuning Complete")
    print("="*60)
    print(f"\nBest Config: {best_config}")
    print(f"Best Validation NDCG@10: {best_score:.4f}")

    # Load and display best model metrics
    if best_exp_dir:
        checkpoint = torch.load(
            os.path.join(best_exp_dir, "best_model.pt"),
            map_location=device
        )
        print(f"\nBest model saved at: {best_exp_dir}")
