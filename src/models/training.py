import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Paths
PROCESSED_DATA_DIR = Path("data/processed_data")
MODELS_DIR = Path("models")

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_scaled.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.pkl"
MODEL_PATH = MODELS_DIR / "rf_model.pkl"


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting model training step")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {MODELS_DIR.resolve()}")

    # Load training data
    logger.info(f"Loading X_train from {X_TRAIN_PATH}")
    X_train = pd.read_csv(X_TRAIN_PATH)

    logger.info(f"Loading y_train from {Y_TRAIN_PATH}")
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    # Load best parameters
    logger.info(f"Loading best parameters from {BEST_PARAMS_PATH}")
    best_params = joblib.load(BEST_PARAMS_PATH)
    logger.info(f"Best parameters: {best_params}")

    # Initialize model
    logger.info("Initializing RandomForestRegressor with best parameters")
    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **best_params
    )

    # Train model
    logger.info("Training model")
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Trained model saved to {MODEL_PATH}")

    logger.info("Model training step completed")


if __name__ == "__main__":
    main()
