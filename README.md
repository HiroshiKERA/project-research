# プロジェクト研究：算術の学習

[環境設定](environment-setup.md)に従って教材をGoogle Driveに置き，[CALT入門](notebooks/calt-tutorial.ipynb)をColabで開きます．
[初日の実験・記録](notebooks/day1.ipynb)，[発展課題](notebooks/extensions.ipynb)へ進みます．

## 編集する場所

| 内容 | ファイル |
|---|---|
| 問題と正解の生成 | [src/addition.py](src/addition.py)の`sum_problem` |
| 入出力の表記 | 同ファイルの`format_problem` |
| 評価指標 | [src/addition_metrics.py](src/addition_metrics.py)の`compute_metrics` |
| データ生成の設定 | [configs/addition/data.yaml](configs/addition/data.yaml) |
| 語彙・token化 | [configs/addition/lexer.yaml](configs/addition/lexer.yaml) |
| モデル・学習・W&B | [configs/addition/train.yaml](configs/addition/train.yaml) |
| 埋め込み | [src/embeddings.py](src/embeddings.py) |

## 実行

データ生成は[src/generate.py](src/generate.py)，学習・評価は[src/train.py](src/train.py)で行います．
設定ファイルはそれぞれの`--config_path`で指定します．
実行コマンドは[scripts/addition.sh](scripts/addition.sh)に記述しています．

ColabでDriveのマウント，教材フォルダへの移動，ライブラリのインストール，W&Bへのログインを行った後，次を実行します．

```python
!bash scripts/addition.sh
```

実験条件を切り替える場合は，PythonファイルとYAMLを編集し，必要に応じて`scripts/`に実行コマンドを追加します．
実験ごとに`train.output_dir`と`train.wandb.name`を変更し，使用したコードと設定を保存してください．
W&Bでは`project`・`group`・`name`が設定と一致しているか確認します．
学習中は`eval/exact_match`と`eval/token_accuracy`，最終評価では`accuracy`を確認します．
生成結果は出力先の`eval_results/`に保存されます．

## 桁数別の評価

[configs/addition/evaluate.yaml](configs/addition/evaluate.yaml)でモデル・表記関数・lexer・桁数を指定し，Colabで実行します．

```python
!bash scripts/evaluate.sh
```

保存済みモデルとtokenizerを読み込むため，再学習は行いません．表記関数とlexerは学習時のものを指定してください．
結果は設定した出力先に保存します．

- `predictions.csv`：各問題の元の桁数，入力，正解，生成結果，完全一致の判定
- `accuracy_by_digits.csv`：桁数の組ごとの件数と完全一致率
- `accuracy_same_digits.csv`：同じ桁数同士の件数と完全一致率
- `length_heatmap.png`：桁数別の完全一致率ヒートマップ
- `length_curve.png`：同じ桁数同士の完全一致率の折れ線

評価・描画コードは実装済みです．[評価Notebook](notebooks/evaluation.ipynb)で設定の確認・評価・図の表示を行えます．
