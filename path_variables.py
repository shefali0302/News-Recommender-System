import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")

# =========================================
# SELECT DATASET HERE
# =========================================
DATASET = "MINDlarge"      # change to "MINDlarge" when needed


# =========================================
# SMALL DATASET PATHS
# =========================================
MIND_SMALL_PATH = os.path.join(DATA_PATH, "MINDsmall")

MIND_SMALL_TRAIN = os.path.join(MIND_SMALL_PATH, "train") # 80% as training set, 20% as validation set
MIND_SMALL_DEV   = os.path.join(MIND_SMALL_PATH, "dev") # as testing set


# =========================================
# LARGE DATASET PATHS
# =========================================
MIND_LARGE_PATH = os.path.join(DATA_PATH, "MINDlarge")

MIND_LARGE_TRAIN = os.path.join(MIND_LARGE_PATH, "train")
MIND_LARGE_DEV   = os.path.join(MIND_LARGE_PATH, "dev")
MIND_LARGE_TEST  = os.path.join(MIND_LARGE_PATH, "test")


# =========================================
# Dynamic Switching
# =========================================
if DATASET == "MINDsmall":
    TRAIN_PATH = MIND_SMALL_TRAIN
    DEV_PATH   = MIND_SMALL_DEV

elif DATASET == "MINDlarge":
    TRAIN_PATH = MIND_LARGE_TRAIN
    DEV_PATH   = MIND_LARGE_DEV

else:
    raise ValueError("Invalid DATASET selected. Choose 'MINDsmall' or 'MINDlarge'.")


# =========================================
# File Paths (used in preprocessing)
# =========================================
TRAIN_NEWS = os.path.join(TRAIN_PATH, "news.tsv")
TRAIN_BEHAVIORS = os.path.join(TRAIN_PATH, "behaviors.tsv")

DEV_NEWS = os.path.join(DEV_PATH, "news.tsv")
DEV_BEHAVIORS = os.path.join(DEV_PATH, "behaviors.tsv")


# =========================================
# Preprocessed Save Location
# =========================================
PREPROCESSED_PATH = os.path.join(DATA_PATH, "preprocessed")
