from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw_data/clean_data.csv")
OUTPUT_DIR = Path("data/processed_data")
PARAMS_PATH = Path("params.yaml")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load parameters
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)["split"]

    test_size = params["test_size"]
    random_state = params["random_state"]
    target_col = params["target_col"]
    drop_columns = params.get("drop_columns", [])

    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Split features and target
    target = df[target_col]
    feats = df.drop([target_col] + drop_columns, axis=1)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        feats,
        target,
        test_size=test_size,
        random_state=random_state
    )

    # Save datasets
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)


if __name__ == "__main__":
    main()
