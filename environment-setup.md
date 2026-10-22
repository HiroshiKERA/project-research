# 環境設定

コードは自分のPCで編集し，Google Colabで実行する．Google Driveでファイルを同期する．

## 1. 教材を準備する

1. [パソコン版Google ドライブ](https://support.google.com/drive/answer/10838124?hl=ja)をインストールし，Colabで使うGoogleアカウントでログインする．
2. 教員が指定したGitHubリポジトリから教材を取得する．
3. マイドライブに`project-research`フォルダを作り，教材を置く．次の配置にする．

```text
マイドライブ/project-research/
├── src/
├── configs/
├── scripts/
└── notebooks/
```

4. VS CodeやCursorでこのフォルダを開く．

## 2. Colabで開く

1. Driveの同期後，`notebooks/calt-tutorial.ipynb`をColabで開く．
2. ランタイムは **GPU・2026.07** を使う（Notebookに設定済み）．
3. Notebookの準備セルを上から実行し，Driveの接続，教材フォルダへの移動，CALTのインストール，W&Bへのログインを行う．

教材フォルダのパスは`/content/drive/MyDrive/project-research`．別の場所に置いた場合は，Notebookの移動先を変更する．

## 3. 編集して実行する

PythonファイルやYAMLをPCで編集・保存し，Driveの同期後，Colabで実行する．
実行するセルとコマンドは各Notebookの指示に従う．
