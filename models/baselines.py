"""
Baseline forecasters for the demand leaderboard.

Every ML project must show the complex model beats simple baselines.
These four models serve as the minimum bar — if LightGBM can't beat
seasonal naive, the feature engineering or training pipeline is broken.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Unified interface for all forecasting models (baselines and ML)."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train on a DataFrame with columns [sku_id, week, demand]."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
          [sku_id, week, p50] (baselines only produce p50)
        """

    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for leaderboard display."""


class NaiveForecaster(BaseForecaster):
    """
    Last-value naive: next week's forecast = this week's demand.
    The simplest possible baseline — anything worse than this is broken.
    """

    def __init__(self):
        self._last_values: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> None:
        latest = (
            df.sort_values("week")
            .groupby("sku_id")["demand"]
            .last()
            .fillna(0)
        )
        self._last_values = latest.to_dict()

    def predict(self, df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
        rows = []
        ref_weeks = df.sort_values("week").groupby("sku_id")["week"].last()
        for sku_id, last_week in ref_weeks.items():
            last_val = self._last_values.get(sku_id, 0)
            for h in range(1, horizon + 1):
                forecast_week = last_week + pd.Timedelta(weeks=h)
                rows.append({"sku_id": sku_id, "week": forecast_week, "p50": last_val})
        return pd.DataFrame(rows)

    def name(self) -> str:
        return "Naive"


class MovingAverageForecaster(BaseForecaster):
    """
    Rolling mean of last N weeks.
    Smooths noise but ignores trend and seasonality.
    """

    def __init__(self, window: int = 4):
        self.window = window
        self._history: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> None:
        for sku_id, grp in df.sort_values("week").groupby("sku_id"):
            vals = grp["demand"].fillna(0).values
            self._history[sku_id] = vals[-self.window :]

    def predict(self, df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
        rows = []
        ref_weeks = df.sort_values("week").groupby("sku_id")["week"].last()
        for sku_id, last_week in ref_weeks.items():
            hist = self._history.get(sku_id, np.array([0.0]))
            avg = float(np.mean(hist)) if len(hist) > 0 else 0.0
            for h in range(1, horizon + 1):
                forecast_week = last_week + pd.Timedelta(weeks=h)
                rows.append({"sku_id": sku_id, "week": forecast_week, "p50": avg})
        return pd.DataFrame(rows)

    def name(self) -> str:
        return f"MovingAverage(w={self.window})"


class SeasonalNaiveForecaster(BaseForecaster):
    """
    Seasonal naive: next week's forecast = same week last year.
    Strong baseline for any data with annual seasonality.
    If we have < 52 weeks of history, falls back to the rolling mean.
    """

    def __init__(self):
        self._history: dict[str, pd.Series] = {}

    def fit(self, df: pd.DataFrame) -> None:
        for sku_id, grp in df.sort_values("week").groupby("sku_id"):
            self._history[sku_id] = grp.set_index("week")["demand"].fillna(0)

    def predict(self, df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
        rows = []
        ref_weeks = df.sort_values("week").groupby("sku_id")["week"].last()
        for sku_id, last_week in ref_weeks.items():
            hist = self._history.get(sku_id, pd.Series(dtype=float))
            for h in range(1, horizon + 1):
                forecast_week = last_week + pd.Timedelta(weeks=h)
                same_week_last_year = forecast_week - pd.Timedelta(weeks=52)
                # Find closest available date if exact match missing
                if len(hist) > 0:
                    delta = abs(hist.index - same_week_last_year)
                    closest_idx = delta.argmin()
                    p50 = float(hist.iloc[closest_idx])
                else:
                    p50 = 0.0
                rows.append({"sku_id": sku_id, "week": forecast_week, "p50": p50})
        return pd.DataFrame(rows)

    def name(self) -> str:
        return "SeasonalNaive"


class ProphetForecaster(BaseForecaster):
    """
    Facebook Prophet — included for completeness.
    Gracefully skipped if Prophet is not installed (common in lightweight envs).
    """

    def __init__(self, yearly_seasonality: bool = True):
        try:
            from prophet import Prophet  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("Prophet not installed — ProphetForecaster will produce zero forecasts.")
            self._available = False
        self._models: dict[str, object] = {}
        self.yearly_seasonality = yearly_seasonality

    def fit(self, df: pd.DataFrame) -> None:
        if not self._available:
            return
        from prophet import Prophet

        for sku_id, grp in df.sort_values("week").groupby("sku_id"):
            prophet_df = grp[["week", "demand"]].rename(columns={"week": "ds", "demand": "y"})
            prophet_df = prophet_df.dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = Prophet(yearly_seasonality=self.yearly_seasonality, weekly_seasonality=False)
                m.fit(prophet_df)
            self._models[sku_id] = m

    def predict(self, df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
        if not self._available:
            ref_weeks = df.sort_values("week").groupby("sku_id")["week"].last()
            rows = [
                {"sku_id": sku_id, "week": last_week + pd.Timedelta(weeks=h), "p50": 0.0}
                for sku_id, last_week in ref_weeks.items()
                for h in range(1, horizon + 1)
            ]
            return pd.DataFrame(rows)

        rows = []
        ref_weeks = df.sort_values("week").groupby("sku_id")["week"].last()
        for sku_id, last_week in ref_weeks.items():
            model = self._models.get(sku_id)
            if model is None:
                for h in range(1, horizon + 1):
                    rows.append({"sku_id": sku_id, "week": last_week + pd.Timedelta(weeks=h), "p50": 0.0})
                continue
            future_dates = [last_week + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]
            future_df = pd.DataFrame({"ds": future_dates})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = model.predict(future_df)
            for _, row in forecast.iterrows():
                rows.append({"sku_id": sku_id, "week": row["ds"], "p50": max(0.0, row["yhat"])})
        return pd.DataFrame(rows)

    def name(self) -> str:
        return "Prophet" if self._available else "Prophet(unavailable)"
