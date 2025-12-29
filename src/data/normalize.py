from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROCESSED_DATA_DIR = Path("data/processed_data")


def main():
    # Load train and test feature sets
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data
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
    X_train_scaled.to_csv(
        PROCESSED_DATA_DIR / "X_train_scaled.csv",
        index=False
    )

    X_test_scaled.to_csv(
        PROCESSED_DATA_DIR / "X_test_scaled.csv",
        index=False
    )


if __name__ == "__main__":
    main()
