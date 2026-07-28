"""
Lightweight Parquet-based feature store.

Avoids train-serve skew by ensuring the same feature transformation code
runs at training time and serving time. Every model is tagged with the
feature version it was trained on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SUPPLIER_REGION_MAP = {"APAC": 0, "EMEA": 1, "NA": 2}
CATEGORY_MAP = {"fast_moving": 0, "intermittent": 1, "seasonal": 2, "slow_moving": 3}


class DemandFeatureStore:
    """
    Versioned feature store for spare-parts demand data.

    Usage:
        store = DemandFeatureStore()
        features = store.generate_features(raw_df, metadata_df)
        store.save(features, version="v1")
        df = store.load("v1")
    """

    def generate_features(
        self, demand: pd.DataFrame, metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute all features from raw demand history + SKU metadata.
        Input demand must have columns: sku_id, week, demand, stockout_flag,
        holiday_flag, promotion_flag.
        """
        df = demand.copy()
        df["week"] = pd.to_datetime(df["week"])
        df = df.sort_values(["sku_id", "week"]).reset_index(drop=True)

        # --- Stockout-adjusted demand ---
        df["demand_adjusted"] = self._impute_stockout(df)

        # --- Lag features ---
        for lag in [1, 2, 4, 8, 12, 26, 52]:
            df[f"lag_{lag}"] = df.groupby("sku_id")["demand_adjusted"].shift(lag)

        # --- Rolling statistics (on adjusted demand) ---
        for window in [4, 12]:
            roll = (
                df.groupby("sku_id")["demand_adjusted"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            df[f"rolling_mean_{window}w"] = roll

            roll_std = (
                df.groupby("sku_id")["demand_adjusted"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )
            df[f"rolling_std_{window}w"] = roll_std.fillna(0)

        # Rolling min/max (4-week window)
        df["rolling_min_4w"] = (
            df.groupby("sku_id")["demand_adjusted"]
            .transform(lambda x: x.shift(1).rolling(4, min_periods=1).min())
        )
        df["rolling_max_4w"] = (
            df.groupby("sku_id")["demand_adjusted"]
            .transform(lambda x: x.shift(1).rolling(4, min_periods=1).max())
        )

        # --- Demand velocity ---
        df["demand_change_1w"] = (
            df.groupby("sku_id")["demand_adjusted"]
            .transform(lambda x: x.shift(1).pct_change(1).replace([np.inf, -np.inf], np.nan))
        )
        df["demand_change_4w"] = (
            df.groupby("sku_id")["demand_adjusted"]
            .transform(lambda x: x.shift(1).pct_change(4).replace([np.inf, -np.inf], np.nan))
        )

        # --- Calendar / seasonality features ---
        df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
        df["month"] = df["week"].dt.month
        df["quarter"] = df["week"].dt.quarter
        df["is_q1"] = (df["quarter"] == 1).astype(int)
        df["is_q4"] = (df["quarter"] == 4).astype(int)

        # --- Intermittency: zero demand ratio over last 12 weeks ---
        df["zero_demand_ratio_12w"] = (
            df.groupby("sku_id")["demand_adjusted"]
            .transform(
                lambda x: x.shift(1).rolling(12, min_periods=1)
                .apply(lambda w: (w == 0).mean(), raw=True)
            )
        )

        # --- Merge SKU metadata ---
        df = df.merge(
            metadata[["sku_id", "category", "supplier_region", "unit_cost", "lead_time_weeks"]],
            on="sku_id",
            how="left",
        )

        # --- Encode categoricals ---
        df["supplier_region_enc"] = df["supplier_region"].map(SUPPLIER_REGION_MAP).fillna(-1).astype(int)
        df["category_enc"] = df["category"].map(CATEGORY_MAP).fillna(-1).astype(int)

        # One-hot encode category
        for cat in CATEGORY_MAP:
            df[f"cat_{cat}"] = (df["category"] == cat).astype(int)

        # One-hot encode supplier region
        for region in SUPPLIER_REGION_MAP:
            df[f"region_{region}"] = (df["supplier_region"] == region).astype(int)

        # Keep holiday/promo flags (already in demand df)
        # Ensure unit_cost is a usable numeric feature
        df["log_unit_cost"] = np.log1p(df["unit_cost"])

        logger.info("Feature generation complete: %d rows, %d columns", len(df), df.shape[1])
        return df

    def _impute_stockout(self, df: pd.DataFrame) -> pd.Series:
        """
        During stockout weeks, observed demand = 0, but true demand is unknown.
        Impute using the rolling mean of non-stockout weeks for that SKU.
        """
        df = df.copy()
        demand_adj = df["demand"].copy()

        for sku_id, grp in df.groupby("sku_id"):
            idx = grp.index
            raw = grp["demand"].values.copy()
            stockout = grp["stockout_flag"].values

            # First fill NaN from missing-value injection with forward-fill
            raw_series = pd.Series(raw).ffill().bfill()
            raw = raw_series.values

            # Compute rolling mean on non-stockout weeks
            adj = raw.copy().astype(float)
            for i in range(len(adj)):
                if stockout[i] == 1:
                    # Use last 12 non-stockout weeks' mean
                    past = [raw[j] for j in range(max(0, i - 12), i) if stockout[j] == 0]
                    adj[i] = np.mean(past) if past else raw[i]

            demand_adj.loc[idx] = adj

        return demand_adj

    def save(self, df: pd.DataFrame, version: str) -> Path:
        path = FEATURES_DIR / f"features_{version}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved features %s → %s (%d rows)", version, path, len(df))
        return path

    def load(self, version: str) -> pd.DataFrame:
        path = FEATURES_DIR / f"features_{version}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature version {version} not found at {path}")
        df = pd.read_parquet(path)
        logger.info("Loaded features %s: %d rows, %d cols", version, len(df), df.shape[1])
        return df

    def get_latest(self) -> pd.DataFrame:
        parquets = sorted(FEATURES_DIR.glob("features_v*.parquet"))
        if not parquets:
            raise FileNotFoundError("No versioned feature files found in data/features/")
        latest = parquets[-1]
        version = latest.stem.replace("features_", "")
        logger.info("Loading latest feature version: %s", version)
        return self.load(version), version

    def get_schema(self) -> dict:
        """Returns feature column → dtype mapping for drift monitoring."""
        parquets = sorted(FEATURES_DIR.glob("features_v*.parquet"))
        if not parquets:
            return {}
        df = pd.read_parquet(parquets[-1], engine="pyarrow").iloc[:0]
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    @property
    def feature_columns(self) -> list[str]:
        """The ordered list of model input features (excludes identifiers and target)."""
        return [
            # Lags
            "lag_1", "lag_2", "lag_4", "lag_8", "lag_12", "lag_26", "lag_52",
            # Rolling stats
            "rolling_mean_4w", "rolling_mean_12w",
            "rolling_std_4w", "rolling_std_12w",
            "rolling_min_4w", "rolling_max_4w",
            # Velocity
            "demand_change_1w", "demand_change_4w",
            # Calendar
            "week_of_year", "month", "quarter", "is_q1", "is_q4",
            # Flags
            "holiday_flag", "promotion_flag",
            # SKU attributes
            "lead_time_weeks", "log_unit_cost",
            # Intermittency
            "zero_demand_ratio_12w",
            # Encodings
            "supplier_region_enc", "category_enc",
            "cat_fast_moving", "cat_slow_moving", "cat_intermittent", "cat_seasonal",
            "region_APAC", "region_EMEA", "region_NA",
        ]
