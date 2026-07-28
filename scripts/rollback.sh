#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${1:?Usage: rollback.sh <version> '<reason>'}"
REASON="${2:?Reason is required}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
python mlops/rollback_model.py --to-version "$VERSION" --reason "$REASON"
