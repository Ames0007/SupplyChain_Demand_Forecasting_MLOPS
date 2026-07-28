"""
Tree-based forecasting models: LightGBM and XGBoost.

These are the production models. Point forecasts used for internal
comparison only; the API serves quantile forecasts (P10/P50/P90).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.baselines import BaseForecaster

logger = logging.getLogger(__name__)

LGBM_DEFAULTS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

XGBOOST_DEFAULTS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}


class LightGBMForecaster(BaseForecaster):
    """Point forecast LightGBM model (RMSE objective)."""

    def __init__(self, feature_cols: list[str], params: dict | None = None):
        import lightgbm as lgb  # noqa: F401 — verify import at construction

        self.feature_cols = feature_cols
        self.params = {**LGBM_DEFAULTS, **(params or {})}
        self._model = None
        self.feature_importances_: pd.Series | None = None

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        import lightgbm as lgb

        X_train = df[self.feature_cols].values
        y_train = df["demand"].values

        fit_kwargs: dict = {}
        if val_df is not None:
            X_val = val_df[self.feature_cols].values
            y_val = val_df["demand"].values
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(self.params["early_stopping_rounds"], verbose=False)]

        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        self._model = lgb.LGBMRegressor(**params)
        self._model.fit(X_train, y_train, **fit_kwargs)

        self.feature_importances_ = pd.Series(
            self._model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)

    def predict(self, df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        X = df[self.feature_cols].values
        preds = np.maximum(0, self._model.predict(X))
        result = df[["sku_id", "week"]].copy()
        result["p50"] = preds
        return result

    def inference_time_ms(self, df: pd.DataFrame) -> float:
        X = df[self.feature_cols].head(100).values
        t0 = time.perf_counter()
        self._model.predict(X)
        return (time.perf_counter() - t0) * 1000

    def name(self) -> str:
        return "LightGBM"


class XGBoostForecaster(BaseForecaster):
    """Point forecast XGBoost model (reg:squarederror objective)."""

    def __init__(self, feature_cols: list[str], params: dict | None = None):
        import xgboost as xgb  # noqa: F401

        self.feature_cols = feature_cols
        self.params = {**XGBOOST_DEFAULTS, **(params or {})}
        self._model = None
        self.feature_importances_: pd.Series | None = None

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        import xgboost as xgb

        X_train = df[self.feature_cols].values
        y_train = df["demand"].values

        fit_kwargs: dict = {}
        if val_df is not None:
            X_val = val_df[self.feature_cols].values
            y_val = val_df["demand"].values
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        params = {k: v for k, v in self.params.items() if k != "early_stopping_rounds"}
        params["early_stopping_rounds"] = self.params["early_stopping_rounds"]
        self._model = xgb.XGBRegressor(**params)
        self._model.fit(X_train, y_train, **fit_kwargs)

        self.feature_importances_ = pd.Series(
            self._model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)

    def predict(self, df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        X = df[self.feature_cols].values
        preds = np.maximum(0, self._model.predict(X))
        result = df[["sku_id", "week"]].copy()
        result["p50"] = preds
        return result

    def name(self) -> str:
        return "XGBoost"


class QuantileLightGBM:
    """
    Quantile regression using LightGBM: trains three separate models for
    P10, P50, P90 (prediction intervals).

    This is the production serving model. The spread P90-P10 gives procurement
    teams a reorder safety buffer without assuming Gaussian residuals — important
    for intermittent and seasonal SKUs.
    """

    QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}

    def __init__(self, feature_cols: list[str], params: dict | None = None):
        self.feature_cols = feature_cols
        self.base_params = {**LGBM_DEFAULTS, **(params or {})}
        self._models: dict[str, object] = {}
        self.feature_importances_: pd.Series | None = None

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        import lightgbm as lgb

        X_train = df[self.feature_cols].values
        y_train = df["demand"].values

        for quantile_name, alpha in self.QUANTILES.items():
            params = {
                **{k: v for k, v in self.base_params.items() if k != "early_stopping_rounds"},
                "objective": "quantile",
                "alpha": alpha,
                "metric": "quantile",
            }
            model = lgb.LGBMRegressor(**params)

            fit_kwargs: dict = {}
            if val_df is not None:
                X_val = val_df[self.feature_cols].values
                y_val = val_df["demand"].values
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(self.base_params["early_stopping_rounds"], verbose=False)
                ]

            model.fit(X_train, y_train, **fit_kwargs)
            self._models[quantile_name] = model
            logger.info("Trained %s model (alpha=%.2f)", quantile_name, alpha)

        # Feature importance from P50 model (representative)
        p50_model = self._models["p50"]
        self.feature_importances_ = pd.Series(
            p50_model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns: sku_id, week, p10, p50, p90."""
        if not self._models:
            raise RuntimeError("QuantileLightGBM has not been trained. Call fit() first.")
        X = df[self.feature_cols].values
        result = df[["sku_id", "week"]].copy()
        for quantile_name, model in self._models.items():
            result[quantile_name] = np.maximum(0, model.predict(X))
        return result

    def inference_time_ms(self, df: pd.DataFrame, n_rows: int = 100) -> float:
        X = df[self.feature_cols].head(n_rows).values
        t0 = time.perf_counter()
        for model in self._models.values():
            model.predict(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return elapsed_ms / n_rows  # per-row latency

    def name(self) -> str:
        return "QuantileLightGBM(P10/P50/P90)"

    @property
    def p50_model(self):
        return self._models.get("p50")
