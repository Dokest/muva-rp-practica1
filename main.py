import os
import numpy as np

from definitions import ROOT_DIR
from src.utils.Trainer import Trainer
from src.utils.utils import load_model

"""
This file executes the saved model and stores the predicted values in Competicion1.txt.
"""


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

# Extract the test data from the datasets
test_dataframes = list(map(Trainer.load_training_file, test_paths))
first_test_dataframe = test_dataframes.pop(0)

test_dataframe_complete = Trainer.merge_datasets(first_test_dataframe, test_dataframes)
test_dataframe_complete = Trainer.remove_id_from_dataframe(test_dataframe_complete)

# Load the trained model
model = load_model("competition_model.pkl")

# Generate the predictions
predicted = model.predict(test_dataframe_complete.to_numpy())

# Save the predictions as txt
np.savetxt("Competicion1.txt", predicted, fmt="%i", delimiter=",")
