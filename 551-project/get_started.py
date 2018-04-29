from data.getData import download
from src.fancy.pkl import pre_and_pickle

download("data/")

pre_and_pickle("data", "test_x", False)
pre_and_pickle("data", "train_x", True, "train_y")