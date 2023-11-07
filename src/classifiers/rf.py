from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classifier_rf():
    return Pipeline([
        ("Random forest", RandomForestClassifier(n_estimators=15))
    ])