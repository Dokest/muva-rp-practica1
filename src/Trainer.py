import typing

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline


class Trainer:
    def __init__(
            self,
            training_paths: [str],
            labels_path: str,
            splitter_func
            ):
        training_dataframes = list(map(self._load_training_file, training_paths))

        complete_dataframe = self._merge_datasets(self._load_labels_file(labels_path), training_dataframes)
        _labels, _undivided_training = self._extract_labels(complete_dataframe)

        if splitter_func:
            self.training, self.testing, self.train_labels, self.test_labels =(
                splitter_func(_undivided_training, _labels))
        else:
            self.training = _undivided_training
            self.train_labels = _labels

    def calculate_scores(self, predicted):
        f1_score = sklearn.metrics.f1_score(self.test_labels, predicted)
        precision = sklearn.metrics.precision_score(self.test_labels, predicted)
        recall = sklearn.metrics.recall_score(self.test_labels, predicted)
        mse = sklearn.metrics.mean_squared_error(self.test_labels, predicted)
        roc = sklearn.metrics.roc_curve(self.test_labels, predicted)

        return {
            "f1": f1_score,
            "precision": precision,
            "recall": recall,
            "mse": mse,
            "roc": roc,
        }

    @staticmethod
    def _load_training_file(training_path: str) -> pd.DataFrame:
        df = pd.read_csv(training_path)
        df = df.rename(columns={"Unnamed: 0": "id"})
        df = df.drop(["0"], axis=1)

        return df

    @staticmethod
    def _load_labels_file(labels_path: str) -> pd.DataFrame:
        df = pd.read_csv(labels_path)
        df = df.rename(columns={"Unnamed: 0": "id"})

        return df

    @staticmethod
    def _merge_datasets(labels_dataset: pd.DataFrame, dataframes: [pd.DataFrame]) -> pd.DataFrame:
        index = 0

        result_dataframe = labels_dataset

        for dataframe in dataframes:
            result_dataframe = result_dataframe.merge(dataframe, on="id", how="right", suffixes=("", "_" + str(index)))
            index += 1

        return result_dataframe

    @staticmethod
    def _extract_labels(dataframe: pd.DataFrame) -> (np.ndarray, np.ndarray):
        training_labels = dataframe[dataframe.columns[1]].to_numpy()
        dataframe = dataframe.drop("malignant", axis=1).to_numpy()

        return training_labels, dataframe
