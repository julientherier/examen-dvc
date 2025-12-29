import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Paths
PROCESSED_DATA_DIR = Path("data/processed_data")
PARAMS_PATH = Path("params.yaml")
MODELS_DIR = Path("models")

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_scaled.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.pkl"


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_params():
    logger.info(f"Loading parameters from {PARAMS_PATH}")
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_estimator_and_grid(cfg: dict):
    model_type = cfg["model_type"]
    logger.info(f"Building estimator for model_type='{model_type}'")

    if model_type == "ridge":
        estimator = Ridge()
        param_grid = {"alpha": cfg["ridge"]["alpha"]}
        logger.info(f"Ridge param grid: {param_grid}")

    elif model_type == "random_forest":
        estimator = RandomForestRegressor(
            random_state=cfg.get("random_state", 42),
            n_jobs=cfg.get("n_jobs", -1),
        )
        rf_cfg = cfg["random_forest"]
        param_grid = {
            "n_estimators": rf_cfg["n_estimators"],
            "max_depth": rf_cfg["max_depth"],
            "min_samples_split": rf_cfg["min_samples_split"],
            "min_samples_leaf": rf_cfg["min_samples_leaf"],
        }
        logger.info(f"RandomForest param grid: {param_grid}")

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return estimator, param_grid


def main():
    logger.info("Starting GridSearch step")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {MODELS_DIR.resolve()}")

    # Load params
    params = _load_params()
    cfg = params["grid_search"]

    cv = cfg.get("cv", 5)
    scoring = cfg.get("scoring", "neg_mean_squared_error")
    n_jobs = cfg.get("n_jobs", -1)
    # <= you can add this in params.yaml if you want
    verbose = cfg.get("verbose", 2)

    logger.info(
        f"GridSearch config — cv={cv}, scoring={scoring}, n_jobs={n_jobs}, verbose={verbose}")

    # Load data
    logger.info(f"Loading X_train from {X_TRAIN_PATH}")
    X_train = pd.read_csv(X_TRAIN_PATH)
    logger.info(f"Loading y_train from {Y_TRAIN_PATH}")
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")  # Series

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    # Model + grid
    estimator, param_grid = _build_estimator_and_grid(cfg)

    # Log how many candidates will be tested
    n_candidates = 1
    for k, v in param_grid.items():
        n_candidates *= len(v)
    logger.info(
        f"GridSearch will evaluate {n_candidates} parameter combinations")
    logger.info(f"Total fits = {n_candidates} * {cv} = {n_candidates * cv}")

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,  # shows progress
    )

    logger.info("Fitting GridSearchCV (this can take some time)...")
    gs.fit(X_train, y_train)
    logger.info("GridSearchCV completed")

    logger.info(f"Best score ({scoring}): {gs.best_score_}")
    logger.info(f"Best params: {gs.best_params_}")

    # Save best params
    joblib.dump(gs.best_params_, BEST_PARAMS_PATH)
    logger.info(f"Saved best params to {BEST_PARAMS_PATH}")

    logger.info("GridSearch step completed")


if __name__ == "__main__":
    main()
