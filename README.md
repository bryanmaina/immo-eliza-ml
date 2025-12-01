
# Property Price Prediction

This project is a machine learning pipeline for predicting property prices based on various features. It uses a structured approach with DVC for pipeline management and MLflow for experiment tracking.

## Technologies Used

- Python 3.11
- Scikit-learn
- Pandas
- XGBoost
- MLflow
- DVC
- Pytest
- Ruff

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r src/requirements.txt
    ```

## Usage

There are two main ways to run the project: using the DVC pipeline or running the training script directly.

### Running Training Directly

You can run the training process directly to experiment with different models and parameters. This script will train multiple models, log them to MLflow, and register the best one.

```bash
python src/training/main.py
```

### Making Predictions

To make predictions using the latest registered model from MLflow:

```bash
python src/prediction/predict.py --data_path data/processed/test.csv
```

## Testing

To run the test suite, use pytest:

```bash
pytest
```

## Code Quality

To check for linting and formatting issues, use ruff:
```bash
ruff check .
```
