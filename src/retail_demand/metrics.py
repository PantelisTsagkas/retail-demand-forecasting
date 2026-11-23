"""Forecast error metrics. All take actuals and predictions as 1-D arrays."""

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error, in units."""
    return float(np.mean(np.abs(predicted - actual)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error, in units. Punishes large misses more than MAE."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Total absolute error as a share of total demand. 0.10 means 10% of volume."""
    return float(np.sum(np.abs(predicted - actual)) / np.sum(actual))


def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared log error. Treats small and large series more evenly."""
    clipped = np.clip(predicted, 0, None)
    return float(np.sqrt(np.mean((np.log1p(clipped) - np.log1p(actual)) ** 2)))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Over (+) or under (-) forecast as a share of total demand."""
    return float((np.sum(predicted) - np.sum(actual)) / np.sum(actual))


ALL_METRICS = {"MAE": mae, "RMSE": rmse, "WAPE": wape, "RMSLE": rmsle, "Bias": bias}


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Every metric for one set of predictions."""
    return {name: fn(actual, predicted) for name, fn in ALL_METRICS.items()}
