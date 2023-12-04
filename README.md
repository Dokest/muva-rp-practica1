# Reconocimiento de patrones: Practica 1
Master Universitario de Visión Artificial \
Universidad Rey Juan Carlos

## Folder structure
The project is structured into folders, each folder contains related code.
- */examples*: code examples and competition code.
- */output*: the model binary that was generated using pickle.
- */src*: main folder of python files
  - */classifiers*: all the classifiers tested
  - */transformers*: all the transformers tested
  - */utils*: all the util that were used including IO, Trainer and our setup function
- */test_data*: contains the test data, only used for the competition
- */training_data*: data used by the model to learn

The files that are the root of the project represent entry points for the application.
That includes:
- *main.py*: Executes the model stored into "./output/competition_model.pkl" and stores the result of the prediction into "Competicion1.txt"
- *competition_train.py*: Trains the model and stores it into the "./output/competition_model.pkl" where it can be later executed from.
- *complete_training.py*: This file represents the method we have used to choose between all the available transformers & classifiers. And it contains all the code to train, predict and then check for overfitting using SearchGridCV.
- *definitions.py*: Is not an entry point, but rather a helper file to make all the paths work from the root of the project.
