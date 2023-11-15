from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

from src.training_setup import init_trainer_for_training

"""
'accuracy': 0.5961199294532628,
'confusion_matrix': array([[227,  67], [162, 111]], 
'f1': 0.49223946784922396,
'precision': 0.6235955056179775,
'recall': 0.4065934065934066
"""

trainer = init_trainer_for_training()
predicted = trainer.run_pipeline(Pipeline([
    ("StandardScaler", StandardScaler()),
    ("MinMax", MinMaxScaler()),
    # ("PCA", PCA(n_components=0.99)),
    ("SVM", SVC(degree=2, C=2, kernel="rbf", gamma="scale")),
]))

print(trainer.calculate_scores(predicted))

"""
{'f1': 0.49223946784922396, 'precision': 0.6235955056179775, 'recall': 0.4065934065934066, 'confusion_matrix': array([[227,  67],
       [162, 111]], dtype=int64), 'mse': 0.4038800705467372, 'roc': (array([0.        , 0.22789116, 1.        ]), array([0.        , 0.40659341, 1.        ]), array([2, 1, 0], dtype=int64))}
   """

"""
{'accuracy': 0.6049382716049383, 'confusion_matrix': array([[213,  81],
       [143, 130]], dtype=int64), 'f1': 0.5371900826446281, 'precision': 0.6161137440758294, 'recall': 0.47619047619047616}
Best Transformer: Pipeline(steps=[('StandardScaler', StandardScaler()),
                ('PCA', PCA(n_components=0.99))])
Best Classifier: SGDClassifier(alpha=2, loss='modified_huber', n_jobs=-1, random_state=1)
"""

"""
{'accuracy': 0.6225749559082893, 'confusion_matrix': array([[229,  65],
       [149, 124]], dtype=int64), 'f1': 0.5367965367965368, 'precision': 0.656084656084656, 'recall': 0.4542124542124542}
Best Transformer: Pipeline(steps=[('StandardScaler', StandardScaler()),
                ('PCA', PCA(n_components=0.99))])
Best Classifier: SVC(C=2, degree=2, random_state=1)
"""