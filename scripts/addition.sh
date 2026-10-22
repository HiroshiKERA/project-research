#!/usr/bin/env bash

# データ生成用のコード
python -m src.generate --config_path configs/addition/data.yaml

# 訓練用のコード
python -m src.train --config_path configs/addition/train.yaml
