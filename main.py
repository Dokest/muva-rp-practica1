import os
import numpy as np

from definitions import ROOT_DIR
from src.utils.Trainer import Trainer
from src.utils.colors import color_red
from src.utils.utils import load_model


"""
This file executes the saved model and stores the predicted values in "Competicion1.txt".
"""

input_model_name = "competition_model.pkl"
input_model_dir = "./output/"

output_prediction_path = "Competicion1.txt"

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

# Test if the model is stored on input_model_path
if not os.path.exists(f"{input_model_dir}{input_model_name}"):
    print(color_red(f"The path \"{input_model_name}\" does not exist because the model was not trained, or it was "
                    f"stored on another folder."))
    exit(1)

# Load the trained model
model = load_model(input_model_name, input_model_dir)

# Generate the predictions
predicted = model.predict(test_dataframe_complete.to_numpy())

# Save the predictions as txt
np.savetxt(output_prediction_path, predicted, fmt="%i", delimiter=",")
