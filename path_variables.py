import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")

# =========================================
# SELECT DATASET HERE
# =========================================
DATASET = "MINDlarge" # change to "MINDlarge" when needed
MODE= "train" # change to "test" when needed

# =========================================
# SMALL DATASET PATHS
# =========================================
MIND_SMALL_PATH = os.path.join(DATA_PATH, "MINDsmall")

MIND_SMALL_TRAIN = os.path.join(MIND_SMALL_PATH, "train") # 80% as training set, 20% as validation set
MIND_SMALL_TEST   = os.path.join(MIND_SMALL_PATH, "dev")

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
    DEV_PATH   = MIND_SMALL_TRAIN  
    TEST_PATH   = MIND_SMALL_TEST

elif DATASET == "MINDlarge":
    TRAIN_PATH = MIND_LARGE_TRAIN
    DEV_PATH   = MIND_LARGE_DEV
    TEST_PATH  = MIND_LARGE_TEST

else:
    raise ValueError("Invalid DATASET selected. Choose 'MINDsmall' or 'MINDlarge'.")


# =========================================
# File Paths (used in preprocessing)
# =========================================
TRAIN_NEWS = os.path.join(TRAIN_PATH, "news.tsv")
TRAIN_BEHAVIORS = os.path.join(TRAIN_PATH, "behaviors.tsv")

DEV_NEWS = os.path.join(DEV_PATH, "news.tsv")
DEV_BEHAVIORS = os.path.join(DEV_PATH, "behaviors.tsv")

TEST_NEWS = os.path.join(TEST_PATH, "news.tsv")
TEST_BEHAVIORS = os.path.join(TEST_PATH, "behaviors.tsv")


# =========================================
# Preprocessed Save Location
# =========================================
PREPROCESSED_PATH = os.path.join(DATA_PATH, "preprocessed")

MIND_SMALL_PREPROCESSED_TRAIN = os.path.join(PREPROCESSED_PATH, "preprocessed_MINDsmall_train_train.pt")
MIND_SMALL_PREPROCESSED_DEV   = os.path.join(PREPROCESSED_PATH, "preprocessed_MINDsmall_train_val.pt")
MIND_SMALL_PREPROCESSED_TEST  = os.path.join(PREPROCESSED_PATH, "preprocessed_MINDsmall_test.pt")

MIND_LARGE_PREPROCESSED_TRAIN = os.path.join(PREPROCESSED_PATH, "preprocessed_MINDlarge_train.pt")
MIND_LARGE_PREPROCESSED_DEV   = os.path.join(PREPROCESSED_PATH, "preprocessed_MINDlarge_val.pt")
MIND_LARGE_PREPROCESSED_TEST  = os.path.join(PREPROCESSED_PATH, "preprocessed_MINDlarge_test.pt")

