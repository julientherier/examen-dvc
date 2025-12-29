import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
PROCESSED_DATA_DIR = Path("data/processed_data")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")

X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_scaled.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"

PREDICTION_PATH = Path("data/prediction.csv")
SCORES_PATH = METRICS_DIR / "scores.json"


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting model evaluation step")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading X_test from {X_TEST_PATH}")
    X_test = pd.read_csv(X_TEST_PATH)

    logger.info(f"Loading y_test from {Y_TEST_PATH}")
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    # Load model
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Predict
    logger.info("Generating predictions")
    y_pred = model.predict(X_test)

    # Save predictions
    logger.info(f"Saving predictions to {PREDICTION_PATH}")
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    pred_df.to_csv(PREDICTION_PATH, index=False)

    # Compute metrics (regression)
    logger.info("Computing evaluation metrics")
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    # Save metrics
    logger.info(f"Saving metrics to {SCORES_PATH}")
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    logger.info(f"Metrics: {scores}")
    logger.info("Model evaluation step completed")


if __name__ == "__main__":
    main()
