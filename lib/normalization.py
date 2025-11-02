"""Utilities for dataset normalization."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _as_2d_column(values) -> np.ndarray:
    """Return the input as a 2D numpy column vector."""

    if hasattr(values, "to_numpy"):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    return np.asarray(arr, dtype=float).reshape(-1, 1)


def normalize_data(
    X,
    T,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Split the dataset, scale features and target, and optionally persist the scalers."""

    X_train, X_test, T_train, T_test = train_test_split(
        X, T, test_size=test_size, random_state=random_state
    )

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    T_train_scaled = y_scaler.fit_transform(_as_2d_column(T_train)).ravel()
    T_test_scaled = y_scaler.transform(_as_2d_column(T_test)).ravel()

    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
        dump(x_scaler, os.path.join(scaler_dir, "x_scaler.joblib"))
        dump(y_scaler, os.path.join(scaler_dir, "y_scaler.joblib"))

    return (
        X_train_scaled,
        X_test_scaled,
        T_train_scaled,
        T_test_scaled,
        x_scaler,
        y_scaler,
    )
