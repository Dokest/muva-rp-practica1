import pandas as pd
import numpy as np


def load_labels() -> pd.DataFrame:
    df = pd.read_csv("training_data/train_label.csv")
    df = df.rename(columns={"Unnamed: 0": "id"})

    return df


def load_training(training_path: str) -> pd.DataFrame:
    df = pd.read_csv(training_path)
    df = df.rename(columns={"Unnamed: 0": "id"})
    df = df.drop(["0"], axis=1)

    return df


training_paths = [
    "training_data/traintab01.csv",
    "training_data/traintab02.csv",
    "training_data/traintab03.csv",
    "training_data/traintab04.csv",
    "training_data/traintab05.csv",
    "training_data/traintab06.csv",
    "training_data/traintab07.csv",
    "training_data/traintab08.csv",
]

training_df: pd.DataFrame = load_labels()
index = 0

for path in training_paths:
    training_df = training_df.merge(load_training(path), on="id", how="right", suffixes=("", "_" + str(index)))
    index += 1

print(training_df)
