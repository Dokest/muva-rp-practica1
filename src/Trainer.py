import typing

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class Trainer:
    def __init__(self, training_paths: [str], labels_path: str):
        self.X_test = None
        self.y_test = None

        training_dataframes = list(map(self._load_training_file, training_paths))

        complete_dataframe = self._merge_datasets(self._load_labels_file(labels_path), training_dataframes)
        complete_dataframe = self._remove_id_from_dataframe(complete_dataframe)

        self.X_train_df, self.y_train_df = self._extract_labels_from_dataframe(complete_dataframe)

        self.X_train = self.X_train_df.to_numpy()
        self.y_train = self.y_train_df.to_numpy()

    def split(self, test_size=0.2, random_state=1):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train,
            self.y_train,
            test_size=test_size,
            random_state=random_state
        )

        return self

    def run_pipeline(self, pipeline: Pipeline, over_data: np.ndarray | None) -> np.ndarray:
        pipeline.fit(self.X_train, self.y_train)

        return pipeline.predict(over_data if over_data is not None else self.X_test)

    def run_bipipeline(self, transform_pipeline: Pipeline, classify_pipeline: Pipeline, over_data: np.ndarray = None):
        pipeline = Pipeline([
            ("transform", transform_pipeline),
            ("classify", classify_pipeline),
        ])

        return self.run_pipeline(pipeline, over_data)

    def calculate_scores(self, predicted):
        f1_score = sklearn.metrics.f1_score(self.y_test, predicted)
        precision = sklearn.metrics.precision_score(self.y_test, predicted)
        recall = sklearn.metrics.recall_score(self.y_test, predicted)
        mse = sklearn.metrics.mean_squared_error(self.y_test, predicted)
        roc = sklearn.metrics.roc_curve(self.y_test, predicted)

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
    def _extract_labels_from_dataframe(dataframe: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        y_train = dataframe[dataframe.columns[0]]
        X_train = dataframe.drop("malignant", axis=1)

        return X_train, y_train

    @staticmethod
    def _remove_id_from_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.drop("id", axis=1)
