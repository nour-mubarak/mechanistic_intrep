#!/usr/bin/env bash
set -e
python -m mechanistic.metrics.bias_metrics --config mechanistic/config.yaml
