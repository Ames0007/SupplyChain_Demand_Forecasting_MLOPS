#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURE_VERSION="${1:-v1}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
echo "Training with feature version: $FEATURE_VERSION"
python training/train.py --feature-version "$FEATURE_VERSION" --model all
