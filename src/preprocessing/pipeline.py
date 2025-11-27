import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.preprocessing.transformers import (
    CorrelationFeatureSelector,
    WinsorizerTransformer,
)


def create_preprocessing_pipeline(
    df: pd.DataFrame, include_feature_selection: bool = True
):
    """
    Creates and returns a scikit-learn preprocessing pipeline.

    This pipeline includes steps for:
    - Imputation of missing values.
    - One-hot encoding of categorical features.
    - Feature selection (low variance and highly correlated).
    - Outlier handling (Winsorization).

    Args:
        df (pd.DataFrame): The input DataFrame to infer feature types from.
    """

    target_variable = "price"
    postal_code_col = "location_postal_code"

    # Dynamically identify numerical and categorical features
    numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Exclude the target variable from numerical features
    if target_variable in numerical_features:
        numerical_features.remove(target_variable)

    # Exclude the target variable from categorical features
    if target_variable in categorical_features:
        categorical_features.remove(target_variable)

    # Handle Postal Code (Force Categorical)
    # Even if pandas sees this as a number (int/float), we want to treat it as a label.
    if postal_code_col in numerical_features:
        numerical_features.remove(postal_code_col)

    # Add to categorical features if it exists in the dataframe but isn't already classified as categorical
    if postal_code_col in df.columns and postal_code_col not in categorical_features:
        categorical_features.append(postal_code_col)

    # Preprocessor for numerical features (imputation, Winsorization)
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "winsorizer",
                WinsorizerTransformer(),
            ),
        ]
    )

    # Preprocessor for categorical features (imputation, one-hot encoding)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown='ignore' ensures we don't crash on new categories in production
            # sparse_output=False is crucial if we want to keep everything as dense pandas DF downstream without extra conversion
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        # Change to 'drop' to ensure 'price' (and any other unspecified column) is removed
        remainder="drop",
        verbose_feature_names_out=True,
    )

    # Build pipeline steps list
    pipeline_steps: list[tuple[str, TransformerMixin]] = [
        ("preprocessor", preprocessor)
    ]

    # Conditionally add feature selection steps
    if include_feature_selection:
        pipeline_steps.extend(
            [
                ("variance_threshold", VarianceThreshold(threshold=0.0)),
                ("correlation_selector", CorrelationFeatureSelector()),
            ]
        )

    # Full pipeline including feature selection steps
    full_pipeline = Pipeline(steps=pipeline_steps)

    # GLOBAL CONFIGURATION: Force the pipeline to output Pandas DataFrames.
    # This ensures that 'X_processed' in your tests (and downstream models)
    # preserves column names, which fixes the shape/name mismatch errors.
    full_pipeline.set_output(transform="pandas")

    return full_pipeline


if __name__ == "__main__":
    # Example usage
    print("Creating a sample DataFrame for testing.")
    sample_data = {
        "property_id": ["A", "B", "C", "D", "E", "F"],
        "location_postal_code": ["1000", "2000", "1000", "3000", "4000", "5000"],
        "location_city": ["Brussels", "Antwerp", "Brussels", "Leuven", "Ghent", "Luik"],
        "area_sqm": [100, 120, np.nan, 90, 80, np.nan],
        "bedrooms": [
            2,
            3,
            2,
            1,
            4,
            3,
        ],
        "price": [
            200000,
            250000,
            180000,
            150000,
            120000,
            100000,
        ],  # Renamed to 'price' to match target logic
    }
    sample_df = pd.DataFrame(sample_data)

    print("Creating a preprocessing pipeline instance with sample DataFrame.")
    pipeline = create_preprocessing_pipeline(sample_df, include_feature_selection=False)

    # Fit and transform
    output = pipeline.fit_transform(sample_df)

    print("Pipeline created successfully.")
    print("Output Shape:", output.shape)
    print(output.head())
    print("Output Columns:", output.columns.tolist())
