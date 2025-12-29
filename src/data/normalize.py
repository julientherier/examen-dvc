import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Paths
PROCESSED_DATA_DIR = Path("data/processed_data")


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting data normalization step")

    # Load train and test feature sets
    logger.info("Loading training and testing feature sets")
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")

    logger.info(
        f"Training features shape: {X_train.shape}, "
        f"Testing features shape: {X_test.shape}"
    )

    # Initialize scaler
    logger.info("Initializing StandardScaler")
    scaler = StandardScaler()

    # Fit on training data only
    logger.info("Fitting scaler on training data")
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data
    logger.info("Transforming test data")
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    # Save scaled datasets
    X_train_scaled_path = PROCESSED_DATA_DIR / "X_train_scaled.csv"
    X_test_scaled_path = PROCESSED_DATA_DIR / "X_test_scaled.csv"

    X_train_scaled.to_csv(X_train_scaled_path, index=False)
    X_test_scaled.to_csv(X_test_scaled_path, index=False)

    logger.info(f"Saved scaled training data to {X_train_scaled_path}")
    logger.info(f"Saved scaled testing data to {X_test_scaled_path}")
    logger.info("Data normalization step completed")


if __name__ == "__main__":
    main()
