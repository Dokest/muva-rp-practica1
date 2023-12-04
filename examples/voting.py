from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.utils.prettify_result import print_pretty_result
from src.utils.training_setup import init_trainer_for_training

# Init training and start the prediction
trainer = init_trainer_for_training()

# Create the transformer
transformers = FeatureUnion([
    ("StandardScaler", StandardScaler()),
    ("MinMax", MinMaxScaler()),
])

# Create the voting classifier with different classifiers
clf = VotingClassifier(voting="soft", estimators=[
    ("gaussian", GaussianNB()),
    ("rf", RandomForestClassifier(n_estimators=50, random_state=1)),
    ("svm", SVC(C=5, kernel="rbf", degree=2, gamma="scale", probability=True)),
    ("decision_tree", DecisionTreeClassifier(criterion="log_loss", random_state=1)),
])

predicted = trainer.fit_and_predict(Pipeline([
    ("transformers", transformers),
    ("classifiers", clf),
]))

# Print the scores
print_pretty_result(trainer.calculate_scores(predicted))
