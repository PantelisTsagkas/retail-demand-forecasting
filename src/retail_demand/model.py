"""Baselines and the LightGBM model."""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_demand.features import FEATURES

logger = logging.getLogger(__name__)

# Each baseline is just a feature column used directly as the forecast.
BASELINES: dict[str, str] = {
    "Last day": "last_1",
    "Same weekday last week": "same_weekday_last_week",
    "Mean of last 4 same weekdays": "same_weekday_mean_4",
}


def new_model(n_estimators: int) -> lgb.LGBMRegressor:
    """LightGBM regressor with the project's fixed settings.

    The objective is squared error on log1p(units), which behaves like RMSLE.
    """
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def fit_with_early_stopping(train: pd.DataFrame, validation: pd.DataFrame) -> lgb.LGBMRegressor:
    """Fit on train, stop when validation error stops improving.

    Args:
        train: Rows with FEATURES and `target`.
        validation: Later rows with the same columns.

    Returns:
        The fitted model. `best_iteration_` holds the chosen number of trees.
    """
    model = new_model(3000)
    model.fit(
        train[list(FEATURES)],
        np.log1p(train["target"]),
        eval_X=validation[list(FEATURES)],
        eval_y=np.log1p(validation["target"]),
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    logger.info("Early stopping chose %d trees", model.best_iteration_)
    return model


def refit(rows: pd.DataFrame, n_estimators: int) -> lgb.LGBMRegressor:
    """Fit a fresh model with a fixed number of trees (train + validation refit).

    Args:
        rows: Rows with FEATURES and `target`.
        n_estimators: Number of trees, taken from early stopping.

    Returns:
        The fitted model.
    """
    model = new_model(n_estimators)
    model.fit(rows[list(FEATURES)], np.log1p(rows["target"]))
    return model


def predict(model: lgb.LGBMRegressor, rows: pd.DataFrame) -> np.ndarray:
    """Predict units, undoing the log transform. Never below zero."""
    raw = np.asarray(model.predict(rows[list(FEATURES)]), dtype=float)
    return np.clip(np.expm1(raw), 0, None)
