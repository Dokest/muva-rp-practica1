from src.classifiers.log_reg import classifier_log_reg
from src.training_setup import init_trainer_for_training
from src.transformers.PCA_Poly import transformer_pca_poly_v1, transformer_pca_poly_v2
from src.classifiers.Knn import classifier_knn

trainer = init_trainer_for_training()

# Process results
predicted = trainer.run_bipipeline(
    transform_pipeline=transformer_pca_poly_v2(),
    classify_pipeline=classifier_knn(),
)

print(trainer.calculate_scores(predicted))
