import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to apply Winsorization to numerical features.
    """

    def __init__(self, lower_bound_quantile=0.01, upper_bound_quantile=0.99):
        if not (0 <= lower_bound_quantile < upper_bound_quantile <= 1):
            raise ValueError(
                "Quantiles must be between 0 and 1, and lower_bound_quantile < upper_bound_quantile."
            )
        self.lower_bound_quantile = lower_bound_quantile
        self.upper_bound_quantile = upper_bound_quantile
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Handle numpy arrays (which come from SimpleImputer) by converting to DF
        # using generic indices if names aren't available.
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Store feature names if available
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()

        for col in X.select_dtypes(include=np.number).columns:
            if not X[col].isnull().all():
                self.lower_bounds[col] = X[col].quantile(self.lower_bound_quantile)
                self.upper_bounds[col] = X[col].quantile(self.upper_bound_quantile)
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_transformed = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_transformed = X.copy()

        for col in X_transformed.select_dtypes(include=np.number).columns:
            # Check if we have bounds for this column (handles generic integer cols or named cols)
            if col in self.lower_bounds and col in self.upper_bounds:
                X_transformed[col] = np.clip(
                    X_transformed[col], self.lower_bounds[col], self.upper_bounds[col]
                )

        # If input was array, return values to maintain compatibility within ColumnTransformer
        if isinstance(X, np.ndarray):
            return X_transformed.values
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.feature_names_in_


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    """
    A custom transformer to remove highly correlated features.
    """

    def __init__(self, correlation_threshold=0.9):
        if not (0 <= correlation_threshold <= 1):
            raise ValueError("Correlation threshold must be between 0 and 1.")
        self.correlation_threshold = correlation_threshold
        self.features_to_drop = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.array(X.columns)

        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Identify columns to drop
        self.features_to_drop = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column] > self.correlation_threshold)
        ]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            # If we get an array here, map it back to the feature names seen in fit
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        return X.drop(columns=self.features_to_drop, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["features_to_drop", "feature_names_in_"])

        if input_features is None:
            input_features = self.feature_names_in_

        return [f for f in input_features if f not in self.features_to_drop]
