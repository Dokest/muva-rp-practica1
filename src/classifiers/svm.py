from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classifier_svm():
    return Pipeline([
        ("Support Vector Machine", svm.SVC())
    ])