import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from src.Trainer import Trainer


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


def k_folds(n_splits: int):
    def _inner_k_folds(data: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        kf = KFold(n_splits=n_splits)

        training = []
        training_lbl = []
        testing = []
        testing_lbl = []

        for _, (train_indices, test_indices) in enumerate(kf.split(data)):
            training = data[train_indices]
            training_lbl = labels[train_indices]

            testing = data[test_indices]
            testing_lbl = labels[test_indices]

        return training, testing, training_lbl, testing_lbl

    return _inner_k_folds


def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=1)


trainer = Trainer(
    training_paths=training_paths,
    labels_path="training_data/train_label.csv",
    splitter_func=split,
)

print("Trainer ready")

preprocess = Pipeline([
    ("StandardScaler", StandardScaler()),
    ("PCA", PCA(n_components=0.90)),
])

features = FeatureUnion([
    ('PolynomialFeatures', PolynomialFeatures(degree=2))
])

postprocess = Pipeline([
    ("PCA", PCA(n_components=0.95)),
    ("LogisticRegression", LogisticRegression(max_iter=10000))
    #("svm", svm.SVC())
])

pipeline = Pipeline([
    ("preprocess", preprocess),
    # ("features", features),
    ("postprocess", postprocess),
])

print("Pipeline complete")

pipeline.fit(trainer.training, trainer.train_labels)

print("Pipeline fitted")

predicted = pipeline.predict(trainer.testing)

print("Complete")

print(trainer.calculate_scores(predicted))
