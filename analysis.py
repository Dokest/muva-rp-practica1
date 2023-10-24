from sklearn.feature_selection import SelectKBest, chi2

from src.training_setup import init_trainer_for_training

###############
### IGNORAR ###
###############

trainer = init_trainer_for_training()
print(trainer.X_train_dataframe)

X_5_best = SelectKBest(k=50).fit(trainer.X_train_dataframe, trainer.y_train_dataframe)

mask = X_5_best.get_support()
new_feat = []

for bool, feature in zip(mask, trainer.X_train_dataframe.columns):
    if bool:
        new_feat.append(feature)


print("The best features are:{}".format(new_feat)) # The list of your 5 best features
