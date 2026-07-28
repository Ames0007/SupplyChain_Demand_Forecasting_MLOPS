# Supply Chain Demand Forecasting · MLOps

> Predicting spare-parts demand across 500 SKUs — with full MLOps infrastructure, drift monitoring, and a production-ready FastAPI serving layer.

---

## 🚀 Live Deployments

| Service | URL |
|---|---|
| 📊 Streamlit Dashboard | [supplychain-forecast-mlops.streamlit.app](https://supplychain-forecast-mlops.streamlit.app) |
| ⚡ FastAPI (Swagger UI) | [supplychain-forecast-api.onrender.com/docs](https://supplychain-forecast-api.onrender.com/docs) |
| 🧪 MLflow Experiment Tracker | [dagshub.com/amitabh1609/SupplyChain_Demand_Forecasting_MLOPS](https://dagshub.com/amitabh1609/SupplyChain_Demand_Forecasting_MLOPS) |

---

## 🛠 Tech Stack

Python 3.11 · LightGBM · XGBoost · PyTorch · MLflow · Evidently AI · FastAPI · Streamlit · Docker · GitHub Actions · DagsHub

---

## The Story Behind This Project

Back in 2023, during my internship at Caterpillar, I built a Random Forest model that identified seasonal demand patterns for spare parts. It was rough around the edges — no versioning, no serving layer, no way to detect when the model started going stale — but it worked well enough that procurement teams actually started using its insights in their planning conversations.

That experience stuck with me. The model was good; the infrastructure around it wasn't. So I rebuilt it from scratch.

This project is that rebuild. Same domain, same problem, but done the way it should have been done the first time: versioned features, a proper model registry, drift detection that fires when something changes in the real world, a REST API that procurement tools could actually call, and a promotion gate so a worse model can never silently replace a better one.

Every design decision here traces back to something that was missing — or that went wrong — in that original prototype.

---

## What It Does

Given a SKU ID and a forecast horizon, the system returns P10/P50/P90 quantile demand forecasts — not just a single number, but a calibrated range that a procurement team can use to set safety stock levels. It also continuously monitors whether the model is still trustworthy, and automatically triggers retraining when it isn't.

```json
POST /forecast
{
  "sku_id": "PART-10482",
  "horizon_weeks": 4,
  "include_intervals": true
}

→ {
    "forecasts": [
      {"week": "2024-06-01", "p10": 82, "p50": 104, "p90": 139},
      {"week": "2024-06-08", "p10": 78, "p50": 99,  "p90": 131}
    ],
    "drift_alert": false,
    "model_version": "v3"
  }
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                     │
│  synthetic_demand.py  →  500 SKUs × 156 weeks                  │
│  4 demand patterns: fast-moving, slow, intermittent, seasonal  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  FEATURE STORE  (Parquet-versioned)                             │
│  33 features: lags, rolling stats, seasonality, stockout adj.  │
│  Every model tagged with the feature version it was trained on  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  TRAINING PIPELINE                                              │
│  Walk-forward CV  →  MLflow tracking  →  Promotion Gate        │
│  Candidate must beat production WAPE by >2% to be promoted     │
│  Registry stages:  None → Staging → Production → Archived      │
└───────────────┬────────────────────────┬────────────────────────┘
                │                        │
   ┌────────────▼──────────┐  ┌──────────▼──────────────────────┐
   │  FastAPI              │  │  Drift Monitor (Evidently AI)   │
   │  POST /forecast       │  │  Data drift   → warn / retrain  │
   │  P10 / P50 / P90      │  │  Pred drift   → KS-test         │
   │  x-model-version hdr  │  │  Perf drift   → rolling WAPE    │
   │  x-drift-alert hdr    │  │  → triggers GitHub Actions      │
   └────────────┬──────────┘  └─────────────────────────────────┘
                │
   ┌────────────▼──────────┐
   │  Streamlit Dashboard  │
   │  6 panels, live data  │
   └───────────────────────┘
```

---

## Results

Trained four model families on a held-out last-13-weeks test set and benchmarked on WAPE, bias, interval coverage, and inference latency — because in production all of those matter.

| Model | WAPE | 80% Coverage | Inference |
|---|---|---|---|
| Seasonal Naive | 65.0% | — | < 1ms |
| Naive (last value) | 58.5% | — | < 1ms |
| Moving Average (4w) | 56.7% | — | < 1ms |
| XGBoost | 37.5% | — | ~8ms |
| **QuantileLightGBM** | **34.9%** | **82.6%** | **~5ms** |

LightGBM beats the best baseline by 46% relative. The 82.6% coverage on the P10–P90 interval slightly exceeds the 80% target, which means the uncertainty estimates are well-calibrated — not too wide, not too narrow.

I also trained an LSTM for comparison and it came in worse than LightGBM on WAPE while being ~10× slower to serve. That result is documented honestly in `models/lstm_model.py`. I'd rather show a controlled experiment that lost than pretend a neural network won.

---

## Why These Design Choices

**Time-based split, not random.** If you randomly shuffle a time-series and split it, future data leaks into training. Your CV numbers look great and your production model fails on day one. Walk-forward cross-validation (train on weeks 1–T, validate on T+1 to T+horizon) is the only approach that mirrors what actually happens in deployment.

**Quantile regression instead of point ± interval.** Spare-parts demand is right-skewed with a hard floor at zero. Assuming symmetric Gaussian uncertainty (±1.96σ) is wrong on both sides. Training separate LightGBM models at α=0.10, 0.50, 0.90 gives asymmetric, empirically calibrated intervals — the procurement team gets a principled safety-stock buffer, not a statistical assumption.

**A promotion gate that can say no.** `promote_model.py` compares the candidate's WAPE to the production model's WAPE and only promotes if the improvement exceeds 2%. If a retrain produces a model that's slightly worse, it gets rejected and logged — the incumbent stays in place. This seems obvious but most ML projects don't have it.

**Two kinds of drift, two different responses.** Data drift (input distributions shift) and concept drift (the relationship between inputs and demand changes) need different treatments. If only features drift, I schedule a retrain. If model performance degrades regardless of feature drift, I trigger an immediate retrain. Conflating them leads to either over-retraining or missing real degradation.

---

## The Drift Demo

This is the part I walk through in interviews. It takes about 3 minutes.

```bash
make bootstrap && make train    # generate data + train production model
make drift-demo                 # inject 2× demand shock on APAC SKUs for 6 weeks
open monitoring/evidently_reports/drift_demo_report.html
```

What happens: 4 out of 6 monitored features drift immediately (KS p ≈ 0). WAPE degrades from 34.9% to ~45.9%. The system responds with `immediate_retrain_and_alert` — both data drift and performance drift triggered simultaneously. You then run `make train && make promote` and watch the retrained model recover.

The APAC shock mirrors a real supply-chain scenario from Caterpillar — a regional disruption that invalidated months of historical lag patterns almost overnight.

---

## API Endpoints

All endpoints live at `https://supplychain-forecast-api.onrender.com`. Every response carries `x-model-version` and `x-drift-alert` headers.

```bash
# Single SKU forecast
curl -X POST https://supplychain-forecast-api.onrender.com/forecast \
  -H "Content-Type: application/json" \
  -d '{"sku_id": "PART-10482", "horizon_weeks": 4}'

# Batch forecast (up to 500 SKUs)
curl -X POST https://supplychain-forecast-api.onrender.com/forecast/batch \
  -H "Content-Type: application/json" \
  -d '{"sku_ids": ["PART-10482", "PART-10483"], "horizon_weeks": 4}'

# Service health + model version
curl https://supplychain-forecast-api.onrender.com/health

# Current production model metadata
curl https://supplychain-forecast-api.onrender.com/model-info

# Latest drift report summary
curl https://supplychain-forecast-api.onrender.com/drift-report
```

Interactive API docs: [supplychain-forecast-api.onrender.com/docs](https://supplychain-forecast-api.onrender.com/docs)

---

## Where the Model Struggles

Being honest about failure modes matters more than hiding them.

**Intermittent SKUs with fewer than 4 weeks of history.** Lag features are NaN. The model falls back to category-level averages, which is better than nothing but not reliable until 12+ weeks of non-zero demand accumulate.

**Stockout periods.** When a part is out of stock, recorded demand is zero — but that's not the same as no demand. The feature store imputes stockout weeks using rolling averages of non-stockout periods, but post-stockout spikes are still systematically under-forecasted.

**New SKUs.** No history means no lag features. Current approach: category-mean fallback. The right fix is similarity-based transfer from the closest historical SKU matched on supplier region, category, and unit cost.

**Sudden supply disruptions.** A regional shock won't appear in any lag feature for 4–8 weeks. Drift monitoring catches this, but the model degrades in the gap before retraining kicks in.

**Region-specific holidays.** The holiday calendar covers major global holidays but not region-specific ones — Chinese New Year for APAC SKUs, for instance. This causes systematic under-forecasting for APAC in weeks 5–7.

---

## Running Locally

```bash
git clone https://github.com/amitabh1609/SupplyChain_Demand_Forecasting_MLOPS.git
cd SupplyChain_Demand_Forecasting_MLOPS

make setup              # pip install -r requirements.txt
make generate-data      # 500 SKUs × 156 weeks
make generate-features  # feature store → features_v1.parquet
make train              # train all models, log to MLflow
make api-local          # FastAPI at localhost:8000
make dashboard-local    # Streamlit at localhost:8501
make drift-demo         # APAC shock demo
```

---

## What's Next

- **Hierarchical forecasting** — reconciling SKU-level forecasts with product-family aggregates top-down and bottom-up
- **Async batch inference** — Celery + Redis for large batches with job-status polling
- **Real-time feature serving** — Redis cache so the API doesn't reload the full Parquet on every request
- **Shadow deployment** — run a candidate model in shadow mode before the promotion gate fires
- **N-BEATS / TFT comparison** — a proper deep-learning comparison with the same rigor as the LSTM experiment
