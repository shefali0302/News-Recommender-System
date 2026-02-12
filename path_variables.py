from pathlib import Path
import os

#BASE_DIR = Path(__name__).resolve().parent
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data")

MINDS_TRAIN= os.path.join(DATA_PATH, "MINDsmall_train" )
MINDS_TRAIN_NEWS = os.path.join(MINDS_TRAIN, "news.tsv")
MINDS_TRAIN_BEHAVIORS = os.path.join(MINDS_TRAIN, "behaviors.tsv")

MINDS_PREPROCESSED_TRAIN = os.path.join(DATA_PATH, "MINDsmall_train_preprocessed.pt")
MINDS_PREPROCESSED_TRAIN_TRAIN= os.path.join(DATA_PATH, "MINDsmall_train_preprocessed_train.pt")
MINDS_PREPROCESSED_TRAIN_VAL = os.path.join(DATA_PATH, "MINDsmall_train_preprocessed_val.pt")