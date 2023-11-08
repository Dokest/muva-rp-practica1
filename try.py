from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classifiers.SVM import svm_classifier
from src.training_setup import init_trainer_for_training
from src.classifiers.log_reg import log_reg_classifier
from src.transformers.PCA import pca_transformer


def generate_param_grid(transf, clfs) -> []:
    grid = []

    for tran in transf:
        for clf in clfs:
            grid.append(tran | clf)

    return grid


# Dummy pipeline -> No tocar
pipeline = Pipeline([
    ('transformer', StandardScaler()),
    ('classifier', LogisticRegression()),
])

# Array with all the transformers available
transformers = [
    pca_transformer(),
    # select_k_best_transformer(), # Raises code warnings
]

# Array with all the classifiers available
classifiers = [
    log_reg_classifier(),
    # random_forest_classifier(),
    # svm_classifier(),
]

param_grid = generate_param_grid(transformers, classifiers)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5)

# Init training and start the prediction
trainer = init_trainer_for_training()
predicted = trainer.run_pipeline(grid_search)

# Print the scores
print(trainer.calculate_scores(predicted))

# Print the best hyperparameters for the transformer and classifier
best_transformer = grid_search.best_estimator_.named_steps['transformer']
best_classifier = grid_search.best_estimator_.named_steps['classifier']

print("Best Transformer:", best_transformer)
print("Best Classifier:", best_classifier)