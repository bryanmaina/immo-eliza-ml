from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from src.preprocessing.transformers import WinsorizerTransformer, CorrelationFeatureSelector

def create_preprocessing_pipeline(df: pd.DataFrame):
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

    target_variable = 'price'
    
    # Dynamically identify numerical and categorical features
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Exclude the target variable from features
    if target_variable in numerical_features:
        numerical_features.remove(target_variable)
    
    # Preprocessor for numerical features (imputation, Winsorization)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorizer', WinsorizerTransformer())
    ])

    # Preprocessor for categorical features (imputation, one-hot encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' ensures we don't crash on new categories in production
        # sparse_output=False is crucial if we want to keep everything as dense pandas DF downstream without extra conversion
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        # Change to 'drop' to ensure 'price' (and any other unspecified column) is removed
        remainder='drop', 
        verbose_feature_names_out=True
    )

    # Full pipeline including feature selection steps
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('correlation_selector', CorrelationFeatureSelector())
    ])

    # GLOBAL CONFIGURATION: Force the pipeline to output Pandas DataFrames.
    # This ensures that 'X_processed' in your tests (and downstream models)
    # preserves column names, which fixes the shape/name mismatch errors.
    full_pipeline.set_output(transform="pandas")

    return full_pipeline

if __name__ == "__main__":
    # Example usage
    print("Creating a sample DataFrame for testing.")
    sample_data = {
        'property_id': ['A', 'B', 'C', 'D'],
        'location_postal_code': ['1000', '2000', '1000', '3000'],
        'location_city': ['Brussels', 'Antwerp', 'Brussels', 'Leuven'],
        'area_sqm': [100, 120, np.nan, 90],
        'bedrooms': [2, 3, 2, 1],
        'price': [200000, 250000, 180000, 150000] # Renamed to 'price' to match target logic
    }
    sample_df = pd.DataFrame(sample_data)

    print("Creating a preprocessing pipeline instance with sample DataFrame.")
    pipeline = create_preprocessing_pipeline(sample_df)
    
    # Fit and transform
    output = pipeline.fit_transform(sample_df)
    
    print("Pipeline created successfully.")
    print("Output Shape:", output.shape)
    print("Output Columns:", output.columns.tolist())