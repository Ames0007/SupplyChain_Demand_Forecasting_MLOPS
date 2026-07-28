# Model Card — Supply Chain Demand Forecasting

**Model:** QuantileLightGBM (P10/P50/P90)
**Version:** Production (see MLflow registry)
**Domain:** Spare-parts demand forecasting
**Owner:** Amitabh Choudhury
**Last updated:** 2026-05-24

---

## Intended Use

**Primary use case:** Forecast weekly spare-parts demand 1–13 weeks ahead for
500+ SKUs, providing point estimates (P50) and uncertainty intervals (P10/P90)
to support procurement planning and safety-stock optimization.

**Out-of-scope uses:**
- Consumer goods demand (different intermittency characteristics)
- Real-time / sub-weekly forecasting (model trained on weekly aggregates)
- New SKUs with zero history (cold-start fallback applies, see Limitations)

---

## Training Data

| Property | Value |
|---|---|
| SKUs | 500 across 4 demand categories |
| History | 156 weeks (3 years) per SKU |
| Training period | Weeks 1–143 (validation: weeks 131–143) |
| Test period | Weeks 144–156 (last 13 weeks withheld) |
| Synthetic data | Reflects Caterpillar SDSA domain: intermittent demand, stockout-adjusted observations, APAC/EMEA/NA regional effects |

**Note on stockout bias:** During stockout weeks, recorded demand = 0 even when
true demand > 0. The feature store imputes stockout weeks using the rolling mean
of non-stockout windows. Models trained on raw stockout data will systematically
under-forecast for SKUs with high stockout frequency.

---

## Evaluation Results

### Overall (held-out test set, 13 weeks)

| Metric | Value |
|---|---|
| WAPE | 34.9% |
| Coverage (80%) | 82.6% |
| vs. best baseline (Moving Average) | 56.7% → 34.9% (-38% relative) |

Note: WAPE of 34.9% reflects a genuinely hard problem — 500 SKUs, 47.5% zero-demand
rate on intermittent SKUs, and a 13-week holdout horizon.

### By SKU Segment

| Category | Relative difficulty | Zero-demand ratio | Notes |
|---|---|---|---|
| fast_moving | easiest | ~6.6% | High volume, regular demand |
| seasonal | second | ~11% | Calendar features capture annual patterns |
| slow_moving | third | ~21% | Low volume amplifies % errors |
| intermittent | hardest | ~47.5% | 60% Bernoulli zero-demand process |

---

## Known Limitations

1. **Intermittent SKUs with <4 weeks of history** — lag features are NaN; model falls back to category-level means. Forecasts will be unreliable.
2. **Demand during stockout periods** — observed demand = 0 does not mean true demand = 0. Imputation is approximate; post-stockout demand may spike unexpectedly.
3. **New SKUs (cold start)** — no lag or rolling features available. Use category average as initial forecast.
4. **Sudden supplier disruptions** — a regional supply shock (e.g., port closure) will not be reflected in any lag feature for 4–8 weeks. Drift monitoring will detect this, but the model will degrade until retraining.
5. **Holiday coverage** — only global industrial holidays modeled. Region-specific public holidays not covered.

---

## Ethical Considerations

Demand forecasts directly influence procurement decisions. Systematic errors have
asymmetric real-world consequences:

- **Under-forecasting** → stockouts → production downtime, customer dissatisfaction
- **Over-forecasting** → excess inventory → working capital lock-up, obsolescence risk

The model is evaluated separately per SKU category precisely because aggregate
accuracy can mask systematic failure on a high-value SKU segment. Intermittent
SKUs — which this model handles worst — are often high-unit-cost parts where
stockout consequences are highest. This limitation is documented explicitly and
should inform safety-stock buffer policy.

---

## Technical Details

| Property | Value |
|---|---|
| Algorithm | LightGBM (quantile objective) |
| Quantiles | P10 (0.10), P50 (0.50), P90 (0.90) |
| Features | 33 engineered features (lags, rolling stats, seasonality, metadata) |
| Hyperparameters | See `models/tree_models.py:LGBM_DEFAULTS` |
| Cross-validation | Walk-forward TimeSeriesSplit (n=5) |
| Random seed | 42 (all models) |
| Training framework | MLflow experiment tracking + Model Registry |
