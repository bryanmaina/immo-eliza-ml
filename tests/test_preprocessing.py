import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import create_preprocessing_pipeline


# Sample DataFrame mimicking the structure of data/raw/data.csv
@pytest.fixture
def sample_raw_data():
    data = {
        "id": [
            34221000,
            2104000,
            34036000,
            58496000,
            48727000,
            22183000,
            13232000,
            48707000,
            74290000,
            58028000,
        ],
        "price": [
            225000.0,
            449000.0,
            335000.0,
            501000.0,
            982700.0,
            548514.0,
            325000.0,
            424000.0,
            185000.0,
            3500000.0,
        ],
        "property_type": [
            "APARTMENT",
            "HOUSE",
            "APARTMENT",
            "HOUSE",
            "DUPLEX",
            "HOUSE",
            "APARTMENT",
            "HOUSE",
            "APARTMENT",
            "HOUSE",
        ],
        "subproperty_type": [
            "APARTMENT",
            "HOUSE",
            "APARTMENT",
            "HOUSE",
            "DUPLEX",
            "HOUSE",
            "APARTMENT",
            "HOUSE",
            "APARTMENT",
            "VILLA",
        ],
        "region": [
            "Flanders",
            "Flanders",
            "Brussels-Capital",
            "Flanders",
            "Wallonia",
            "Flanders",
            "Wallonia",
            "Flanders",
            "Wallonia",
            "Flanders",
        ],
        "province": [
            "Antwerp",
            "East Flanders",
            "Brussels",
            "Antwerp",
            "Walloon Brabant",
            "Flemish Brabant",
            "Walloon Brabant",
            "East Flanders",
            "Liège",
            "West Flanders",
        ],
        "locality": [
            "Antwerp",
            "Gent",
            "Brussels",
            "Turnhout",
            "Nivelles",
            "Halle-Vilvoorde",
            "Nivelles",
            "Gent",
            "Liège",
            "Brugge",
        ],
        "zip_code": [2050, 9185, 1070, 2275, 1410, 1700, 1420, 9800, 4000, 8300],
        "latitude": [
            51.2171725,
            51.174944,
            50.8420431,
            51.2383125,
            np.nan,
            50.9194375,
            50.6836074,
            50.9830147,
            50.6318255,
            51.3533226,
        ],
        "longitude": [
            4.3799821,
            3.845248,
            4.3345427,
            4.8171921,
            np.nan,
            4.932433,
            4.3715129,
            3.5707416,
            5.5573521,
            3.2963532,
        ],
        "construction_year": [
            1963.0,
            np.nan,
            1968.0,
            2024.0,
            2022.0,
            1970.0,
            1980.0,
            2023.0,
            np.nan,
            1935.0,
        ],
        "total_area_sqm": [
            100.0,
            120.0,
            142.0,
            187.0,
            169.0,
            187.0,
            80.0,
            155.0,
            120.0,
            277.0,
        ],
        "surface_land_sqm": [
            np.nan,
            680.0,
            np.nan,
            505.0,
            np.nan,
            710.0,
            np.nan,
            291.0,
            0.0,
            588.0,
        ],
        "nbr_frontages": [2.0, np.nan, 2.0, np.nan, 2.0, 4.0, 2.0, 4.0, 2.0, 6.0],
        "nbr_bedrooms": [2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 1.0, 4.0, 2.0, 6.0],
        "equipped_kitchen": [
            "INSTALLED",
            "MISSING",
            "INSTALLED",
            "MISSING",
            "HYPER_EQUIPPED",
            "MISSING",
            "MISSING",
            "HYPER_EQUIPPED",
            "MISSING",
            "MISSING",
        ],
        "fl_furnished": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "fl_open_fire": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "fl_terrace": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "terrace_sqm": [5.0, 0.0, np.nan, 0.0, 20.0, 0.0, 11.0, 0.0, np.nan, 0.0],
        "fl_garden": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "garden_sqm": [0.0, 0.0, 0.0, 0.0, 142.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "fl_swimming_pool": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "fl_floodzone": [0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        "state_building": [
            "MISSING",
            "MISSING",
            "AS_NEW",
            "MISSING",
            "AS_NEW",
            "AS_NEW",
            "MISSING",
            "GOOD",
            "MISSING",
            "TO_RENOVATE",
        ],
        "primary_energy_consumption_sqm": [
            231.0,
            221.0,
            np.nan,
            99.0,
            19.0,
            221.0,
            np.nan,
            np.nan,
            212.0,
            394.0,
        ],
        "epc": [
            "C",
            "C",
            "MISSING",
            "A",
            "A+",
            "MISSING",
            "MISSING",
            "MISSING",
            "C",
            "D",
        ],
        "heating_type": [
            "GAS",
            "MISSING",
            "GAS",
            "MISSING",
            "GAS",
            "MISSING",
            "GAS",
            "MISSING",
            "MISSING",
            "MISSING",
        ],
        "fl_double_glazing": [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
        "cadastral_income": [
            922.0,
            406.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    }
    return pd.DataFrame(data)


def test_pipeline_output_shape_and_target_exclusion(sample_raw_data):
    """
    Test that the pipeline returns a DataFrame with the correct shape and
    that the target variable 'price' is excluded from the processed features.
    """
    df = sample_raw_data.copy()
    pipeline = create_preprocessing_pipeline(df)

    # Fit and transform the data
    X_processed = pipeline.fit_transform(df)

    # Note: X_processed is now guaranteed to be a DataFrame due to pipeline.set_output(transform="pandas")

    # Check that 'price' column is not in the processed features
    assert "price" not in X_processed.columns, (
        "Target variable 'price' should be excluded from processed features."
    )

    assert X_processed.shape[1] > 0, "Processed DataFrame should have features."
    assert X_processed.shape[0] == df.shape[0], "Number of rows should remain the same."


def test_imputation_handling_numerical_and_categorical(sample_raw_data):
    """
    Test that the preprocessing pipeline correctly handles missing values.
    """
    df = sample_raw_data.copy()

    # Introduce some NaNs
    df.loc[0, "total_area_sqm"] = np.nan
    df.loc[1, "property_type"] = np.nan

    pipeline = create_preprocessing_pipeline(df)
    X_processed = pipeline.fit_transform(df)

    assert not X_processed.isnull().any().any(), (
        "Processed DataFrame should not contain NaN values."
    )


def test_one_hot_encoding_categorical_features(sample_raw_data):
    """
    Test that the preprocessing pipeline correctly applies one-hot encoding.
    """
    df = sample_raw_data.copy()
    pipeline = create_preprocessing_pipeline(df)
    X_processed = pipeline.fit_transform(df)

    initial_categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Check that original categorical columns are removed
    for col in initial_categorical_cols:
        if col != "price":
            assert col not in X_processed.columns, (
                f"Original categorical column '{col}' should be removed."
            )

    feature_names = X_processed.columns.tolist()

    # Check for existence of some encoded columns.
    # We use 'in' rather than 'startswith' to be robust to varying Scikit-Learn naming conventions
    # (e.g. presence or absence of 'cat__' prefix or other name modifications).
    assert any("property_type_HOUSE" in col for col in feature_names), (
        "One-hot encoded 'property_type_HOUSE' should be present."
    )
    assert any("equipped_kitchen_INSTALLED" in col for col in feature_names), (
        "One-hot encoded 'equipped_kitchen_INSTALLED' should be present."
    )


def test_winsorization_outlier_handling(sample_raw_data):
    """
    Test that the WinsorizerTransformer correctly caps extreme values.
    """
    df = sample_raw_data.copy()

    # Introduce clear outliers
    df.loc[0, "total_area_sqm"] = 1000000.0  # Upper outlier
    df.loc[1, "total_area_sqm"] = 0.1  # Lower outlier

    pipeline = create_preprocessing_pipeline(df)
    X_processed = pipeline.fit_transform(df)

    # The pipeline outputs named features now, so we can find them directly
    # However, exact names depend on if they were renamed by ColumnTransformer
    processed_total_area_col = [
        col for col in X_processed.columns if "total_area_sqm" in col
    ][0]

    # Access internal transformer
    winsorizer = pipeline.named_steps["preprocessor"].named_transformers_["num"][
        "winsorizer"
    ]

    # Use raw column name for lookup in winsorizer bounds
    lower_bound = winsorizer.lower_bounds["total_area_sqm"]
    upper_bound = winsorizer.upper_bounds["total_area_sqm"]

    assert all(X_processed[processed_total_area_col] >= lower_bound - 1e-9), (
        "Values should be capped at lower bound."
    )
    assert all(X_processed[processed_total_area_col] <= upper_bound + 1e-9), (
        "Values should be capped at upper bound."
    )


def test_correlation_feature_selection(sample_raw_data):
    """
    Test that CorrelationFeatureSelector removes one of two highly correlated features.
    """
    df = sample_raw_data.copy()

    # Create two highly correlated numerical features
    df["highly_correlated_feature"] = (
        df["total_area_sqm"] * 1.001 + np.random.rand(len(df)) * 0.001
    )

    # Fill NaNs before pipeline so correlation logic works on clean data
    df["total_area_sqm"] = df["total_area_sqm"].fillna(df["total_area_sqm"].median())
    df["highly_correlated_feature"] = df["highly_correlated_feature"].fillna(
        df["highly_correlated_feature"].median()
    )

    pipeline = create_preprocessing_pipeline(df)
    X_processed = pipeline.fit_transform(df)

    # Find the processed feature names
    processed_total_area_col_names = [
        col for col in X_processed.columns if "total_area_sqm" in col
    ]
    processed_correlated_col_names = [
        col for col in X_processed.columns if "highly_correlated_feature" in col
    ]

    # After correlation selection, exactly one of them should be present in the FINAL output
    total_area_present = len(processed_total_area_col_names) > 0
    correlated_feature_present = len(processed_correlated_col_names) > 0

    assert (total_area_present and not correlated_feature_present) or (
        not total_area_present and correlated_feature_present
    ), "Exactly one of the highly correlated features should be dropped."


def test_variance_threshold_feature_selection(sample_raw_data):
    """
    Test that VarianceThreshold correctly removes low-variance features.
    """
    df = sample_raw_data.copy()

    # Add a low variance feature
    df["low_variance_feature"] = 5

    pipeline = create_preprocessing_pipeline(df)
    X_processed = pipeline.fit_transform(df)

    # The feature should be dropped from the final dataframe
    processed_low_variance_col = [
        col for col in X_processed.columns if "low_variance_feature" in col
    ]
    assert len(processed_low_variance_col) == 0, (
        "Low variance feature should be removed."
    )
