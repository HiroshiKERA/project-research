#!/usr/bin/env bash
set -e
python -m src.train_extensions --config_path "$1"
python -m src.evaluate_extensions --config_path "$1"
