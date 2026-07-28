#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
REGION="${1:-APAC}"
MULTIPLIER="${2:-2.0}"
WEEKS="${3:-6}"
echo "Simulating drift: ${MULTIPLIER}x demand shock on ${REGION} for ${WEEKS} weeks"
python mlops/drift_simulator.py \
  --shock-region "$REGION" \
  --shock-multiplier "$MULTIPLIER" \
  --duration-weeks "$WEEKS"
echo "Drift report: monitoring/evidently_reports/drift_demo_report.html"
