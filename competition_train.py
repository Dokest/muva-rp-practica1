import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

from definitions import ROOT_DIR
from src.utils.Trainer import Trainer
from src.utils.utils import save_model


"""
This file is used to train the model. After training, the model is saved into "output_path.pkl"
"""

output_model_path = "competition_model.pkl"
output_model_dir = "./output/"

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

# Init trainer with all the training paths
trainer = Trainer(
    training_paths=training_paths,
    labels_path=os.path.join(ROOT_DIR, "training_data/train_label.csv"),
)

# Create the pipeline
pipeline = Pipeline([
    ("StandardScaler", StandardScaler()),
    ("MinMax", MinMaxScaler()),
    ("SVM", SVC(degree=2, C=5, kernel="rbf", gamma="scale", random_state=1)),
])

# Fit the model
pipeline = pipeline.fit(trainer.X_train, trainer.y_train)

# Store it in the file system as a Pickle
save_model(pipeline, output_model_path, output_model_dir)
