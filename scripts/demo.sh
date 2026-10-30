#!/usr/bin/env bash
set -euo pipefail
python -m src.mlops.training --trials "${1:-10}"
python -m src.mlops.drift
