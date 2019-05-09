import dataset
import os

DATASETS_FOLDER = "datasets"
DATASET_NAME = "toy"

dataset_dictionary = {
        "name": DATASET_NAME,
        "path": os.path.join(DATASETS_FOLDER, DATASET_NAME, "dirty.csv"),
        "clean_path": os.path.join(DATASETS_FOLDER, DATASET_NAME, "clean.csv")
    }

d = dataset.Dataset(dataset_dictionary)
