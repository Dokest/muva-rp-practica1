import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.classifiers.log_reg import log_reg_classifier
from src.transformers.PCA import pca_transformer
from src.training_setup import init_trainer_for_training


class TransformedTargetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, transformer=None, classifier=None):
        self.transformer = transformer
        self.classifier = classifier

    def fit(self, X, y):
        if self.transformer:
            y_transformed = self.transformer.fit(y)
        else:
            y_transformed = y  # No transformation

        self.classifier.fit(X, y_transformed)
        return self

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        #if self.transformer:
        #    y_pred = self.transformer.inverse(y_pred)
        return y_pred


trainer = init_trainer_for_training()

transformer = pca_transformer()
classifier = log_reg_classifier()

pipe = Pipeline([
    ("transformer", PCA(n_components=0.95)),
    ("classifier", LogisticRegression())
])

clf = GridSearchCV(pipe, [
    {
        "transformer": [PCA(n_components=0.95)],
        "classifier": [LogisticRegression(max_iter=10000)],
    },
], verbose=2, cv=5, scoring=sklearn.metrics.accuracy_score)

trainer.run_pipeline(clf)


exit(0)

print(classifier | transformer)

clf = GridSearchCV(pipe, [
    transformer | classifier,
], verbose=2, scoring=sklearn.metrics.accuracy_score)

"""
clf = GridSearchCV(Pipeline([
    ("transformer", transformer),
    ("classifier", classifier),
]), t_params | clf_params)
"""


predicted = trainer.run_pipeline(clf)

print(clf.best_params_)
print(trainer.calculate_scores(predicted))
