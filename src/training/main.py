import os
import subprocess
import time

import dvc.api
import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow import log_artifact, log_param
from mlflow.models import evaluate, infer_signature
from mlflow.sklearn import log_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.preprocessing.pipeline import create_preprocessing_pipeline


def train_and_log_models():
    """
    Loads data, splits it, trains multiple regression models, evaluates them using
    cross-validation and a final test set, logs results to MLflow, and registers
    the best model.
    """
    print("MLflow run started. This is a placeholder for actual training logic.")

    # -------------------------------------------------------------------------
    # 1. Configuration Loading (DVC)
    # -------------------------------------------------------------------------
    params = {}
    try:
        params = dvc.api.params_show()
        print(f"DVC parameters: {params}")
    except Exception as e:
        print(
            f"Warning: Could not load DVC parameters. Proceeding with defaults. Error: {e}"
        )

    # -------------------------------------------------------------------------
    # 2. Data Loading, Preprocessing, and Splitting
    # -------------------------------------------------------------------------
    raw_data_path = "data/raw/data.csv"
    print(f"Loading raw data from {raw_data_path}")

    X_processed_df = None
    y_raw = None
    test_split_ratio = params.get("test_split_ratio", 0.2)
    random_seed = params.get("random_seed", 42)

    try:
        raw_data = pd.read_csv(raw_data_path)

        # Separate target variable before preprocessing
        X_raw = raw_data.drop("price", axis=1)
        y_raw = raw_data["price"]

        # Create and fit preprocessing pipeline
        preprocessing_pipeline = create_preprocessing_pipeline(X_raw)
        X_processed = preprocessing_pipeline.fit_transform(
            X_raw, y_raw
        )  # Pass y_raw to fit for consistency => I am not sure if this is the best approach

        # Convert back to DataFrame for train_test_split (handling array output)
        if isinstance(X_processed, np.ndarray):
            # For simplicity, assign generic column names after transformation
            num_cols = X_processed.shape[1]
            X_processed_df = pd.DataFrame(
                X_processed, columns=[f"feature_{i}" for i in range(num_cols)]
            )
        else:
            X_processed_df = X_processed

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

    # Exit if data was not loaded successfully
    if X_processed_df is None or y_raw is None:
        return

    # -------------------------------------------------------------------------
    # 3. Model Training and Evaluation
    # -------------------------------------------------------------------------
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
        print(f"\nTraining and evaluating {name}...")

        # Use a nested run for each model for better organization in MLflow
        with mlflow.start_run(run_name=name, nested=True):
            log_param("model_name", name)
            log_param("test_split_ratio", test_split_ratio)
            log_param("random_seed", random_seed)

            # Evaluate the model
            trained_model, test_r2 = evaluate_with_cv(
                model, X_train, y_train, X_test, y_test, cv_folds=5, name=name
            )

            # Store results for the parent run to find the best model
            trained_models[name] = trained_model
            model_r2_scores[name] = test_r2

    # -------------------------------------------------------------------------
    # 4. Identify and Register the Best Model
    # -------------------------------------------------------------------------
    if model_r2_scores:
        # Get the name of the model with the highest R2 score
        best_model_name = max(model_r2_scores, key=lambda name: model_r2_scores[name])
        best_model = trained_models[best_model_name]
        print(
            f"\nBest model based on Test R-squared: {best_model_name} (R2: {model_r2_scores[best_model_name]:.4f})"
        )

        # Log the best model in the parent run
        with mlflow.start_run(run_name=f"Best Model: {best_model_name}", nested=True):
            signature = infer_signature(X_test, best_model.predict(X_test))
            # Keep log_model here, as it is preferred for model registry interaction
            log_model(
                sk_model=best_model,
                name="best_model",
                registered_model_name="BelgianPropertyPricePredictor",
                signature=signature,
                input_example=X_test.iloc[[0]],
            )
            print(
                f"Best model '{best_model_name}' registered as 'BelgianPropertyPricePredictor'."
            )

    # -------------------------------------------------------------------------
    # 5. Save and Log the Preprocessing Pipeline to be used for prediction
    # -------------------------------------------------------------------------
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    pipeline_path = f"{model_dir}/preprocessing_pipeline.pkl"
    joblib.dump(preprocessing_pipeline, pipeline_path)
    log_artifact(pipeline_path, "preprocessing")

    print("MLflow run ended.")


def evaluate_with_cv(model, X, y, X_test, y_test, cv_folds=5, name=""):
    """
    Evaluate model with cross-validation and final test evaluation for regression.

    This function now uses mlflow.sklearn.log_model, which logs the model directly
    to the MLflow artifact store.
    """
    print(f"Evaluating {name} with cross-validation...")

    # 1. Cross-Validation
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    # Log CV results
    mlflow.log_metrics({"cv_mean_r2": cv_scores.mean(), "cv_std_r2": cv_scores.std()})
    print(f"{name} CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Re-train on the full training data
    model.fit(X, y)

    # 2. Log Model Artifact using log_model
    artifact_path = f"{name}_model_cv"

    signature = infer_signature(X_test, model.predict(X_test))

    # Use log_model to log directly to the artifact store
    model_info = log_model(
        sk_model=model,
        name=artifact_path,
        signature=signature,
        input_example=X_test.iloc[[0]],
    )

    # Prepare evaluation data for mlflow.evaluate
    eval_data = X_test.copy()
    eval_data[y_test.name] = y_test.values

    # Run MLflow evaluation
    result = evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets=y_test.name,
        model_type="regressor",
        evaluator_config={"log_model_explainability": False},
    )

    # Log metrics from the evaluation result
    test_r2 = result.metrics["r2_score"]
    mlflow.log_metrics(result.metrics)

    # Compare CV and test performance
    cv_r2 = cv_scores.mean()

    mlflow.log_metrics(
        {
            "test_r2_score": test_r2,
            "cv_vs_test_r2_diff": abs(cv_r2 - test_r2),
        }
    )
    print(f"{name} Test R2: {test_r2:.4f}")

    return model, test_r2


if __name__ == "__main__":
    # Placeholder for MLflow run
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    MLFLOW_TRACKING_DB = "sqlite:///mlflow.db"

    # -------------------------------------------------------------------------
    # MLflow UI Management (Starts UI and keeps script running until Ctrl+C)
    # -------------------------------------------------------------------------
    # Define the command to start the MLfl
    mlflow_command = ["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_DB]
    mlflow_process = None

    print("\nTraining completed. Attempting to start MLflow UI...")

    try:
        # Start the MLflow UI as a separate, non-blocking process
        # We redirect stdout/stderr to DEVNULL to prevent it from cluttering the console
        mlflow_process = subprocess.Popen(
            mlflow_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"MLflow UI started in background (PID: {mlflow_process.pid}).")
        print("Access the UI (if running locally) at: http://127.0.0.1:5000")
        print("Press Ctrl+C to stop the script and terminate the MLflow UI.")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_DB)
        mlflow.set_experiment("immo-elisa-ml-experiment-cv")
        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run():
            train_and_log_models()

        # Keep the main process running indefinitely until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping MLflow UI...")

    finally:
        # Ensure the MLflow UI process is terminated when the script exits
        if mlflow_process and mlflow_process.poll() is None:
            mlflow_process.terminate()
            mlflow_process.wait()
            print("MLflow UI process terminated successfully.")

        print("Program finished.")
