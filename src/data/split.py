from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw_data/clean_data.csv")
OUTPUT_DIR = Path("data/processed")
TARGET_COL = "silica_concentrate"


def main():
    # Create output directory if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Split features and target
    target = df[TARGET_COL]
    feats = df.drop([TARGET_COL], axis=1)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        feats,
        target,
        test_size=0.3,
        random_state=42
    )

    # Save datasets
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)


if __name__ == "__main__":
    main()
