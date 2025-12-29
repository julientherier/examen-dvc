import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Paths
RAW_DATA_PATH = Path("data/raw_data/clean_data.csv")
OUTPUT_DIR = Path("data/processed_data")
PARAMS_PATH = Path("params.yaml")


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting data split step")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processed data directory: {OUTPUT_DIR.resolve()}")

    # Load parameters
    logger.info(f"Loading split parameters from {PARAMS_PATH}")
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["split"]

    test_size = params["test_size"]
    random_state = params["random_state"]
    target_col = params["target_col"]
    drop_columns = params.get("drop_columns", [])

    logger.info(
        f"Split configuration — test_size={test_size}, "
        f"random_state={random_state}, target_col={target_col}, "
        f"drop_columns={drop_columns}"
    )

    # Load dataset
    logger.info(f"Loading raw dataset from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Dataset loaded with shape: {df.shape}")

    # Split features and target
    target = df[target_col]
    feats = df.drop([target_col] + drop_columns, axis=1)

    logger.info(
        f"Features shape: {feats.shape}, Target shape: {target.shape}"
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        feats,
        target,
        test_size=test_size,
        random_state=random_state
    )

    logger.info(
        f"Train shapes — X: {X_train.shape}, y: {y_train.shape}"
    )
    logger.info(
        f"Test shapes  — X: {X_test.shape}, y: {y_test.shape}"
    )

    # Save datasets
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    logger.info("Split datasets successfully saved")
    logger.info("Data split step completed")


if __name__ == "__main__":
    main()
