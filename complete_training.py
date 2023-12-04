from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers.adaboost import adaboost_classifier
from src.classifiers.svm import svm_classifier
from src.utils.colors import color_green
from src.utils.permute import generate_param_grid
from src.utils.prettify_result import print_pretty_result
from src.utils.training_setup import init_trainer_for_training
from src.transformers.Standard_MinMax import standard_minmax_transformer


"""
This file is the training file that we have used for the most part of the training.

It contains all the elements necessary to:
    - Read the training data & the test data
    - Run a GridSearchCV over permutations of transformers & classifiers
    - Train the GridSearchCV model
    - Print the test results
    - Print the training results to check for overfitting
"""


# Dummy pipeline -> No tocar
pipeline = Pipeline([
    ('transformer', StandardScaler()),
    ('classifier', LogisticRegression()),
])

# Array with all the transformers to try
transformers = [
    # pca_transformer(),
    # polynomial_features_with_pca_transformer(),
    # select_k_best_transformer(), # Raises code warnings
    standard_minmax_transformer(),
]

# Array with all the classifiers to try
classifiers = [
    # log_reg_classifier(),
    # random_forest_classifier(),
    # adaboost_classifier(),
    svm_classifier(C=[5], kernel=["rbf"], degree=[2], gamma=["scale"]),
]

# Permute all combinations of transformers and classifiers
param_grid = generate_param_grid(transformers, classifiers)

# Create model with permutations and k-folds
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring="accuracy", cv=5, verbose=3)

# Init training and start the prediction
trainer = init_trainer_for_training()
print(color_green(">> Init training"))
predicted = trainer.fit_and_predict(grid_search)

# Print the test scores
print(color_green(">> Test scores"))
print_pretty_result(trainer.calculate_scores(predicted))

# Print the best hyperparameters for the transformer and classifier
best_transformer = grid_search.best_estimator_.named_steps['transformer']
best_classifier = grid_search.best_estimator_.named_steps['classifier']

print("Best Transformer:", best_transformer)
print("Best Classifier:", best_classifier)

# Check for overfitting by comparing it with the prediction over the training data
print(color_green(">> Checking overfitting"))
predicted_train = trainer.fit_and_predict(grid_search, over_data=trainer.X_train)
print_pretty_result(trainer.calculate_scores(predicted_train, trainer.y_train))  # Check for overfitting in the training data
