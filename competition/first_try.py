import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

from definitions import ROOT_DIR
from src.Trainer import Trainer

# Entrenar con todos los datos
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

trainer = Trainer(
    training_paths=training_paths,
    labels_path=os.path.join(ROOT_DIR, "training_data/train_label.csv"),
)

test_paths = [
    os.path.join(ROOT_DIR, "test_data/testtab01.csv"),
    os.path.join(ROOT_DIR, "test_data/testtab02.csv"),
    os.path.join(ROOT_DIR, "test_data/testtab03.csv"),
    os.path.join(ROOT_DIR, "test_data/testtab04.csv"),
    os.path.join(ROOT_DIR, "test_data/testtab05.csv"),
    os.path.join(ROOT_DIR, "test_data/testtab06.csv"),
    os.path.join(ROOT_DIR, "test_data/testtab07.csv"),
    os.path.join(ROOT_DIR, "test_data/testtab08.csv"),
]

test_dataframes = list(map(Trainer.load_training_file, test_paths))
first_test_dataframe = test_dataframes.pop(0)

test_dataframe_complete = Trainer.merge_datasets(first_test_dataframe, test_dataframes)
test_dataframe_complete = Trainer.remove_id_from_dataframe(test_dataframe_complete)

# Hacer la clasificación
pipeline = Pipeline([
    ("StandardScaler", StandardScaler()),
    ("MinMax", MinMaxScaler()),
    ("SVM", SVC(degree=2, C=2, kernel="rbf", gamma="scale", random_state=1)),
])

predicted = trainer.fit_and_predict(pipeline, test_dataframe_complete.to_numpy())

# Pasarlo a txt
predicted_str = '\n'.join(str(x) for x in predicted)

with open("./competition/Competicion1_grupo_D_hugo_jorge.txt", "w") as file:
    file.write(predicted_str)
