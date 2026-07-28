#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Generating synthetic demand data..."
python data/generator/synthetic_demand.py
echo "Done. Files written to data/raw/"
