"""
Synthetic spare-parts demand generator.

Simulates enterprise supply-chain complexity: intermittent demand,
stockout-adjusted observations, seasonal patterns, promotional lifts,
and supplier lead-time variability across 500 SKUs over 3 years.

This mirrors the domain from the Caterpillar SDSA prototype (2023),
rebuilt here with full ML infrastructure.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)

RAW_DIR = Path(__file__).parent.parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

N_SKUS = 500
N_WEEKS = 156  # 3 years
CATEGORIES = ["fast_moving", "slow_moving", "intermittent", "seasonal"]
SUPPLIER_REGIONS = ["APAC", "EMEA", "NA"]

# Regional demand multipliers — small but real, mirroring procurement patterns
REGION_MULTIPLIERS = {"APAC": 0.95, "EMEA": 1.05, "NA": 1.00}

# Category base demand ranges (units/week)
CATEGORY_BASE_DEMAND = {
    "fast_moving":  (80, 200),
    "slow_moving":  (5, 30),
    "intermittent": (2, 20),
    "seasonal":     (20, 100),
}

# Category intermittency: probability of non-zero demand in a given week
ZERO_DEMAND_PROB = {
    "fast_moving":  0.02,  # almost always has demand
    "slow_moving":  0.25,
    "intermittent": 0.60,  # 60% chance of zero
    "seasonal":     0.10,
}


def _generate_sku_metadata(n_skus: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    sku_ids = [f"PART-{10000 + i}" for i in range(n_skus)]
    # Balanced but unequal category distribution
    cat_weights = [0.35, 0.25, 0.25, 0.15]
    categories = rng.choice(CATEGORIES, size=n_skus, p=cat_weights)
    supplier_regions = rng.choice(SUPPLIER_REGIONS, size=n_skus)
    unit_costs = np.round(rng.uniform(10, 2500, size=n_skus), 2)
    lead_time_weeks = rng.integers(1, 9, size=n_skus)  # 1–8 weeks

    return pd.DataFrame({
        "sku_id": sku_ids,
        "category": categories,
        "supplier_region": supplier_regions,
        "unit_cost": unit_costs,
        "lead_time_weeks": lead_time_weeks,
    })


def _generate_calendar(n_weeks: int) -> pd.DataFrame:
    """Generate week-level calendar with holiday and promotion flags."""
    rng = np.random.default_rng(99)
    start = pd.Timestamp("2021-01-04")  # Monday
    weeks = pd.date_range(start=start, periods=n_weeks, freq="W-MON")

    # Holidays: ~8% of weeks (roughly 12 per year)
    holiday_flag = (rng.random(n_weeks) < 0.08).astype(int)
    # Cluster holidays around known industrial holiday periods (weeks 51-52, 1-2, 26)
    holiday_weeks = list(range(0, 3)) + list(range(25, 27)) + list(range(50, 53))
    for w in holiday_weeks:
        if w < n_weeks:
            holiday_flag[w] = 1

    # Promotions: ~12% of weeks, clustered in Q2/Q4
    promotion_flag = np.zeros(n_weeks, dtype=int)
    for year_start in [0, 52, 104]:
        q2_start = year_start + 13
        q4_start = year_start + 39
        for window_start in [q2_start, q4_start]:
            for w in range(window_start, min(window_start + 8, n_weeks)):
                if rng.random() < 0.5:
                    promotion_flag[w] = 1

    return pd.DataFrame({"week": weeks, "holiday_flag": holiday_flag, "promotion_flag": promotion_flag})


def _generate_sku_demand(
    sku_id: str,
    category: str,
    supplier_region: str,
    lead_time_weeks: int,
    calendar: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(calendar)

    # Base demand level for this SKU
    lo, hi = CATEGORY_BASE_DEMAND[category]
    base_demand = rng.uniform(lo, hi)

    # --- Trend: slight linear drift (±15% over 3 years) ---
    trend_slope = rng.uniform(-0.0005, 0.0005)  # per week
    trend = 1.0 + trend_slope * np.arange(n)

    # --- Seasonality: SKU-specific amplitude + phase ---
    amplitude = rng.uniform(0.05, 0.40) if category != "intermittent" else rng.uniform(0.02, 0.15)
    phase = rng.uniform(0, 2 * np.pi)
    if category == "seasonal":
        amplitude = rng.uniform(0.30, 0.70)

    t = np.arange(n)
    seasonality = 1.0 + amplitude * np.sin(2 * np.pi * t / 52.0 + phase)

    # --- Regional multiplier ---
    region_mult = REGION_MULTIPLIERS[supplier_region]

    # --- Base signal ---
    demand_signal = base_demand * trend * seasonality * region_mult

    # --- Demand spikes: 3–5 per year ---
    n_spikes_per_year = int(rng.integers(3, 6))
    total_spikes = n_spikes_per_year * 3
    spike_weeks = rng.choice(n, size=min(total_spikes, n), replace=False)
    spike_multipliers = rng.uniform(2.0, 5.0, size=len(spike_weeks))
    spike_signal = np.ones(n)
    for w, m in zip(spike_weeks, spike_multipliers):
        # Spikes decay over 2 weeks
        spike_signal[w] *= m
        if w + 1 < n:
            spike_signal[w + 1] *= 1 + (m - 1) * 0.4

    demand_signal *= spike_signal

    # --- Holiday effect: -25% demand ---
    holiday_mult = np.where(calendar["holiday_flag"].values == 1, 0.75, 1.0)
    demand_signal *= holiday_mult

    # --- Promotion effect: +35% demand ---
    promo_mult = np.where(calendar["promotion_flag"].values == 1, 1.35, 1.0)
    demand_signal *= promo_mult

    # --- Intermittency (Bernoulli zero-demand process) ---
    zero_prob = ZERO_DEMAND_PROB[category]
    is_zero = rng.random(n) < zero_prob
    demand_signal = np.where(is_zero, 0.0, demand_signal)

    # --- Additive noise ---
    noise_scale = base_demand * 0.08
    noise = rng.normal(0, noise_scale, size=n)
    demand_signal = np.maximum(0, demand_signal + noise)

    # --- Round to integers (spare parts are discrete) ---
    true_demand = np.round(demand_signal).astype(int)

    # --- Stockout periods: 2–4 windows of 2–4 weeks each ---
    observed_demand = true_demand.copy()
    stockout_flag = np.zeros(n, dtype=int)
    n_stockout_windows = int(rng.integers(2, 5))
    for _ in range(n_stockout_windows):
        start_w = int(rng.integers(0, n - 4))
        duration = int(rng.integers(2, 5))
        end_w = min(start_w + duration, n)
        observed_demand[start_w:end_w] = 0
        stockout_flag[start_w:end_w] = 1

    # --- Inject 2–3% missing values ---
    n_missing = int(np.ceil(0.025 * n))
    missing_idx = rng.choice(n, size=n_missing, replace=False)
    observed_demand_float = observed_demand.astype(float)
    observed_demand_float[missing_idx] = np.nan

    df = pd.DataFrame({
        "sku_id": sku_id,
        "week": calendar["week"],
        "demand": observed_demand_float,
        "true_demand": true_demand.astype(float),
        "stockout_flag": stockout_flag,
        "holiday_flag": calendar["holiday_flag"].values,
        "promotion_flag": calendar["promotion_flag"].values,
    })
    return df


def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Generating SKU metadata for %d SKUs ...", N_SKUS)
    metadata = _generate_sku_metadata(N_SKUS)
    calendar = _generate_calendar(N_WEEKS)

    logger.info("Generating %d weeks of demand history per SKU ...", N_WEEKS)
    demand_frames = []
    for _, row in metadata.iterrows():
        rng = np.random.default_rng(abs(hash(row["sku_id"])) % (2**32))
        df_sku = _generate_sku_demand(
            sku_id=row["sku_id"],
            category=row["category"],
            supplier_region=row["supplier_region"],
            lead_time_weeks=int(row["lead_time_weeks"]),
            calendar=calendar,
            rng=rng,
        )
        demand_frames.append(df_sku)

    demand_history = pd.concat(demand_frames, ignore_index=True)
    logger.info(
        "Generated demand_history: %d rows, %.1f%% missing",
        len(demand_history),
        demand_history["demand"].isna().mean() * 100,
    )

    # Save outputs
    demand_path = RAW_DIR / "demand_history.csv"
    meta_path = RAW_DIR / "sku_metadata.csv"
    demand_history.drop(columns=["true_demand"]).to_csv(demand_path, index=False)
    metadata.to_csv(meta_path, index=False)

    # Also save true_demand for drift simulator use
    demand_history.to_csv(RAW_DIR / "demand_history_with_true.csv", index=False)

    logger.info("Saved demand_history.csv (%d rows) → %s", len(demand_history), demand_path)
    logger.info("Saved sku_metadata.csv (%d rows) → %s", len(metadata), meta_path)

    _print_summary(demand_history, metadata)
    return demand_history, metadata


def _print_summary(demand: pd.DataFrame, metadata: pd.DataFrame) -> None:
    merged = demand.merge(metadata[["sku_id", "category"]], on="sku_id")
    print("\n=== Dataset Summary ===")
    print(f"  Total rows:     {len(demand):,}")
    print(f"  SKUs:           {demand['sku_id'].nunique()}")
    print(f"  Date range:     {demand['week'].min()} → {demand['week'].max()}")
    print(f"  Missing demand: {demand['demand'].isna().mean():.2%}")
    print(f"  Zero demand:    {(demand['demand'] == 0).mean():.2%}")
    print(f"\n  By category:")
    for cat, grp in merged.groupby("category"):
        mean_d = grp["demand"].mean()
        zero_pct = (grp["demand"] == 0).mean()
        print(f"    {cat:<20} mean={mean_d:6.1f}  zero={zero_pct:.1%}")
    print()


if __name__ == "__main__":
    generate_dataset()
