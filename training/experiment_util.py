import os
import datetime
import json

def create_experiment_folder(config_dict, base_dir="results"):
    now = datetime.datetime.now()
    folder_name = now.strftime("%d-%m-%Y_%H-%M-%S")

    exp_path = os.path.join(base_dir, folder_name)
    os.makedirs(exp_path, exist_ok=True)

    # Save config as JSON
    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    print(f"Experiment folder created at: {exp_path}")
    return exp_path
