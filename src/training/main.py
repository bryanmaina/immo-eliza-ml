import os

import dvc.api
import joblib  # To save preprocessing pipeline and models
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow import log_artifact, log_metric, log_param
from mlflow.models import evaluate, infer_signature
from mlflow.sklearn import log_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Assuming src/preprocessing/pipeline.py contains create_preprocessing_pipeline
from src.preprocessing.pipeline import create_preprocessing_pipeline


def train_and_log_models():
    print("MLflow run started. This is a placeholder for actual training logic.")
    # Placeholder for DVC parameters
    params = {}
    try:
        params = dvc.api.params_show()
        print(f"DVC parameters: {params}")
    except Exception as e:
        print(
            f"Warning: Could not load DVC parameters. Proceeding with defaults. Error: {e}"
        )

    # Data loading and splitting (T019)
    raw_data_path = "data/raw/data.csv"
    print(f"Loading preprocessed data from {raw_data_path}")
    try:
        # Load the raw data to create and fit the pipeline,
        # as the pipeline expects the original features.
        raw_data = pd.read_csv(raw_data_path)

        # Separate target variable before preprocessing
        X_raw = raw_data.drop("price", axis=1)
        y_raw = raw_data["price"]

        # Create and fit preprocessing pipeline
        preprocessing_pipeline = create_preprocessing_pipeline(
            X_raw, include_feature_selection=False
        )  # Pass raw features to infer types
        X_processed = preprocessing_pipeline.fit_transform(X_raw)

        # Reconstruct DataFrame after preprocessing for train_test_split
        # Note: Column names might change due to one-hot encoding, etc.
        # For now, we'll use a simplified approach; proper column tracking will be added later if needed.
        # Convert processed numpy array back to DataFrame for easier handling
        # This step requires careful handling of column names. For now, assume a simple case.
        # As a placeholder, let's create a dummy DataFrame.
        # In a real scenario, ColumnTransformer's get_feature_names_out would be used.
        if isinstance(X_processed, np.ndarray):
            # Placeholder for column names after preprocessing
            # This needs to be correctly handled by getting feature names from ColumnTransformer
            # For now, let's assume it's just numerical for splitting
            X_processed_df = pd.DataFrame(X_processed)
        else:
            X_processed_df = X_processed

        test_split_ratio = params.get("test_split_ratio", 0.2)
        random_seed = params.get("random_seed", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed_df,
            y_raw,
            test_size=test_split_ratio,
            random_state=random_seed,
        )
        print(
            f"Data loaded and split: X_train shape {X_train.shape}, X_test shape {X_test.shape}"
        )

    except FileNotFoundError:
        print(f"Error: {raw_data_path} not found. Please ensure raw data is available.")
        exit(1)
    except KeyError:
        print(
            "Error: 'price' column not found. Ensure target variable is correctly named."
        )
        exit(1)

    # Model training and evaluation (T020)
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(random_state=random_seed),
        "XGBoostRegressor": xgb.XGBRegressor(
            random_state=random_seed, eval_metric="rmse"
        ),
    }

    trained_models = {}
    model_r2_scores = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}, MAE: {mae:.2f}")

        trained_models[name] = model
        model_r2_scores[name] = r2

        # Log parameters and metrics (T021)
        log_param(f"{name}_test_split_ratio", test_split_ratio)
        log_metric(f"{name}_rmse", rmse)
        log_metric(f"{name}_r2", r2)
        log_metric(f"{name}_mae", mae)

        # Log the model artifact (T021)
        log_model(
            sk_model=model,
            name=f"{name}_model",
            input_example=X_test.iloc[[0]],
        )

    # Identify and register the best model (T022)
    if model_r2_scores:
        best_model_name = max(model_r2_scores, key=lambda name: model_r2_scores[name])
        best_model = trained_models[best_model_name]
        print(
            f"\nBest model based on R-squared: {best_model_name} (R2: {model_r2_scores[best_model_name]:.2f})"
        )

        # Register the best model in MLflow Model Registry
        log_model(
            sk_model=best_model,
            name="best_model",
            registered_model_name="BelgianPropertyPricePredictor",
        )
        print(
            f"Best model '{best_model_name}' registered as 'BelgianPropertyPricePredictor'."
        )

    # Save the preprocessing pipeline (as it will be needed for prediction)
    # We need to save the pipeline after fitting it to the raw data.
    # This will be logged as an artifact of the parent run.
    # We assume 'model' directory exists or is created by DVC.
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)  # Ensure 'model' directory exists
    joblib.dump(preprocessing_pipeline, f"{model_dir}/preprocessing_pipeline.pkl")
    log_artifact(f"{model_dir}/preprocessing_pipeline.pkl", "preprocessing")

    print("MLflow run ended.")


def evaluate_with_cv(model, X, y, eval_data, cv_folds=5):
    """
    This is defenitely function stolen from MLflow guides!
    Evaluate model with cross-validation and final test evaluation."""

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

    # Log CV results
    mlflow.log_metrics({"cv_mean_f1": cv_scores.mean(), "cv_std_f1": cv_scores.std()})

    # Train on full dataset
    model.fit(X, y)

    # Final evaluation
    signature = infer_signature(X, model.predict(X))
    log_model(model, name="model", signature=signature)
    model_uri = mlflow.get_artifact_uri("model")

    result = evaluate(model_uri, eval_data, targets="label", model_type="classifier")

    # Compare CV and test performance
    test_f1 = result.metrics["f1_score"]
    cv_f1 = cv_scores.mean()

    mlflow.log_metrics(
        {
            "cv_vs_test_diff": abs(cv_f1 - test_f1),
            "potential_overfit": cv_f1 - test_f1 > 0.05,
        }
    )


if __name__ == "__main__":
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("immo-elisa-ml")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run():
        train_and_log_models()
