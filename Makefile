.PHONY: generate-data generate-features train promote up down drift-demo rollback test lint

PYTHON = python
MLFLOW_URI = https://dagshub.com/amitabh1609/SupplyChain_Demand_Forecasting_MLOPS.mlflow

generate-data:
	$(PYTHON) data/generator/synthetic_demand.py

generate-features:
	$(PYTHON) -c "\
	import pandas as pd; \
	from features.feature_store import DemandFeatureStore; \
	demand = pd.read_csv('data/raw/demand_history.csv', parse_dates=['week']); \
	meta = pd.read_csv('data/raw/sku_metadata.csv'); \
	store = DemandFeatureStore(); \
	features = store.generate_features(demand, meta); \
	store.save(features, 'v1'); \
	print('Features saved as v1')"

train:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) training/train.py --feature-version v1 --model all

promote:
	@echo "Usage: make promote RUN_ID=<run-id>"
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) training/promote_model.py --run-id $(RUN_ID) --metric wape

drift-demo:
	$(PYTHON) mlops/drift_simulator.py --shock-region APAC --shock-multiplier 2.0 --duration-weeks 6

rollback:
	@echo "Usage: make rollback VERSION=<version> REASON='<reason>'"
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) mlops/rollback_model.py --to-version $(VERSION) --reason "$(REASON)"

up:
	docker-compose up --build -d
	@echo ""
	@echo "Services started:"
	@echo "  MLflow:    http://localhost:5001"
	@echo "  API:       http://localhost:8000"
	@echo "  Dashboard: http://localhost:8501"
	@echo "  Reports:   http://localhost:8502"

down:
	docker-compose down

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	ruff check . --fix

setup:
	pip install -r requirements.txt

bootstrap: generate-data generate-features
	@echo "Bootstrap complete. Run 'make train' to train models."

api-local:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard-local:
	streamlit run dashboard/app.py

mlflow-local:
	mlflow server --host 0.0.0.0 --port 5001 \
	  --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlruns
