from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    ("PCA", PCA(n_components=0.99)),
    ("SVM", SVC(degree=2)),
]))

print(trainer.calculate_scores(predicted))

"""
{'f1': 0.49223946784922396, 'precision': 0.6235955056179775, 'recall': 0.4065934065934066, 'confusion_matrix': array([[227,  67],
       [162, 111]], dtype=int64), 'mse': 0.4038800705467372, 'roc': (array([0.        , 0.22789116, 1.        ]), array([0.        , 0.40659341, 1.        ]), array([2, 1, 0], dtype=int64))}
   """