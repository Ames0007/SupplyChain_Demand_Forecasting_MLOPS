"""
Walk-forward cross-validation for time-series models.

Uses TimeSeriesSplit to simulate production deployment: the model always
trains on past data and predicts future data. Random splits would leak
future information into training — a common mistake that inflates reported
accuracy and produces models that fail in production.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from training.evaluate import compute_metrics, compute_segment_metrics

logger = logging.getLogger(__name__)


def walk_forward_cv(
    model_cls,
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    model_params: dict | None = None,
    log_to_mlflow: bool = False,
) -> dict:
    """
    Walk-forward cross-validation using time-ordered splits.

    For each split:
      - Train on weeks 1..T
      - Validate on weeks T+1..T+horizon
      - Compute metrics and log fold results

    Returns cv_summary with mean ± std across folds.
    """
    # Sort by week and get unique sorted time indices
    df = df.sort_values("week").reset_index(drop=True)
    # Get the ordered unique week positions for splitting
    unique_weeks = df["week"].sort_values().unique()
    n_weeks = len(unique_weeks)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    week_indices = np.arange(n_weeks)

    fold_metrics: list[dict] = []

    for fold_num, (train_week_idx, val_week_idx) in enumerate(tscv.split(week_indices)):
        train_weeks = set(unique_weeks[train_week_idx])
        val_weeks = set(unique_weeks[val_week_idx])

        train_df = df[df["week"].isin(train_weeks)].copy()
        val_df = df[df["week"].isin(val_weeks)].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        logger.info(
            "Fold %d/%d — train: %d rows (%s→%s), val: %d rows (%s→%s)",
            fold_num + 1,
            n_splits,
            len(train_df),
            train_df["week"].min().date(),
            train_df["week"].max().date(),
            len(val_df),
            val_df["week"].min().date(),
            val_df["week"].max().date(),
        )

        # Instantiate model fresh for each fold
        model = model_cls(feature_cols=feature_cols, params=model_params)
        model.fit(train_df, val_df=val_df)

        # Get predictions on validation set
        preds = model.predict(val_df)
        val_merged = val_df[["sku_id", "week", "demand"]].merge(preds, on=["sku_id", "week"], how="left")

        # Handle quantile model output vs. point model output
        if "p50" in val_merged.columns:
            y_pred = val_merged["p50"].values
        else:
            logger.warning("Fold %d: No p50 column found in predictions", fold_num + 1)
            continue

        y_true = val_merged["demand"].values
        y_p10 = val_merged.get("p10", pd.Series(dtype=float)).values if "p10" in val_merged else None
        y_p90 = val_merged.get("p90", pd.Series(dtype=float)).values if "p90" in val_merged else None

        metrics = compute_metrics(y_true, y_pred, y_p10, y_p90)
        metrics["fold"] = fold_num + 1
        fold_metrics.append(metrics)

        if log_to_mlflow:
            _log_fold_to_mlflow(fold_num + 1, metrics)

        logger.info("Fold %d WAPE=%.2f%% MAE=%.2f", fold_num + 1, metrics["wape"], metrics["mae"])

    if not fold_metrics:
        return {}

    cv_summary = _summarize_folds(fold_metrics)
    logger.info(
        "CV complete — mean WAPE=%.2f%% ±%.2f%%",
        cv_summary["wape_mean"],
        cv_summary["wape_std"],
    )
    return cv_summary


def _summarize_folds(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std for each metric across CV folds."""
    keys = [k for k in fold_metrics[0].keys() if k != "fold"]
    summary = {}
    for key in keys:
        values = [f[key] for f in fold_metrics if not np.isnan(f.get(key, np.nan))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        else:
            summary[f"{key}_mean"] = np.nan
            summary[f"{key}_std"] = np.nan
    summary["n_folds"] = len(fold_metrics)
    return summary


def _log_fold_to_mlflow(fold_num: int, metrics: dict) -> None:
    try:
        import mlflow
        for metric_name, value in metrics.items():
            if metric_name != "fold" and not np.isnan(float(value)):
                mlflow.log_metric(f"fold_{fold_num}_{metric_name}", value, step=fold_num)
    except Exception as exc:
        logger.warning("MLflow fold logging failed: %s", exc)
