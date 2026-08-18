import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ReplaceInvalidValues(BaseEstimator, TransformerMixin):
    """Заменяет известные некорректные значения на NaN."""

    def __init__(self):
        self.late_columns = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Некорректный возраст
        X["age"] = X["age"].replace(0, np.nan)

        # Специальные значения в признаках просрочек
        for column in self.late_columns:
            X[column] = X[column].replace([96, 98], np.nan)

        return X


preprocessor = Pipeline([
    ("invalid_values", ReplaceInvalidValues()),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
