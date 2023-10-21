import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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

training_labels = training_df[training_df.columns[1]].to_numpy()
training_np: np = training_df.drop("malignant", axis=1).to_numpy()

# Split training
kf = KFold(n_splits=5)
kf.get_n_splits(training_np)


def everything(training, training_lbl, testing, testing_lbl, scaler, dimRed, classifier) -> []:
    # Reduce dimension
    scaler.fit(training)

    training = scaler.transform(training)
    testing = scaler.transform(testing)

    dimRed.fit(training)

    training = dimRed.transform(training)
    testing = dimRed.transform(testing)

    # Classify (predict)
    classifier.fit(training, training_lbl)

    # Result
    return classifier.predict(testing)


def calculate_scores(real, predicted):
    f1_score = sklearn.metrics.f1_score(real, predicted)
    precision = sklearn.metrics.precision_score(real, predicted)
    recall = sklearn.metrics.recall_score(real, predicted)
    mse = sklearn.metrics.mean_squared_error(real, predicted)
    roc = sklearn.metrics.roc_curve(real, predicted)

    print({
        "f1": f1_score,
        "precision": precision,
        "recall": recall,
        "mse": mse,
        "roc": roc,
    })


for i, (train_indices, test_indices) in enumerate(kf.split(training_np)):
    print(f"Fold: {i}")

    training = training_np[train_indices]
    training_lbl = training_labels[train_indices]

    testing = training_np[test_indices]
    testing_lbl = training_labels[test_indices]

    scaler = StandardScaler()
    pca = PCA(n_components=0.85)
    logisticRegr = LogisticRegression()

    predicted = everything(
        training=training,
        training_lbl=training_lbl,
        testing=testing,
        testing_lbl=testing_lbl,
        scaler=scaler,
        dimRed=pca,
        classifier=logisticRegr,
    )

    calculate_scores(testing_lbl, predicted)

    sklearn.metrics.RocCurveDisplay.from_predictions(testing_lbl, predicted)
    plt.show()

"""
training = training_np[train_indices]
training_lbl = training_labels[train_indices]

testing = training_np[test_indices]
testing_lbl = training_labels[test_indices]

# Reduce dimension
scaler = StandardScaler()
scaler.fit(training)

training = scaler.transform(training)
testing = scaler.transform(testing)

pca = PCA(n_components=0.85)
pca.fit(training)

training = pca.transform(training)
testing = pca.transform(testing)

# Classify (predict)
logisticRegr = LogisticRegression()
logisticRegr.fit(training, training_lbl)

# Result
x = logisticRegr.score(testing, testing_lbl)
print(x)
"""
