import os

from definitions import ROOT_DIR
from src.utils.Trainer import Trainer

# Define the paths for the different training datasets
training_paths = [
    os.path.join(ROOT_DIR, "training_data/traintab01.csv"),
    os.path.join(ROOT_DIR, "training_data/traintab02.csv"),
    os.path.join(ROOT_DIR, "training_data/traintab03.csv"),
    os.path.join(ROOT_DIR, "training_data/traintab04.csv"),
    os.path.join(ROOT_DIR, "training_data/traintab05.csv"),
    os.path.join(ROOT_DIR, "training_data/traintab06.csv"),
    os.path.join(ROOT_DIR, "training_data/traintab07.csv"),
    os.path.join(ROOT_DIR, "training_data/traintab08.csv"),
]


# Read training data
def init_trainer_for_training() -> Trainer:
    """
    Helper function that creates a Trainer instance with the training paths and the labels path.
    """
    return Trainer(
        training_paths=training_paths,
        labels_path=os.path.join(ROOT_DIR, "training_data/train_label.csv"),
    ).split()
