"""足し算のモデルを学習・評価する。"""
from transformers import set_seed
import click  # CLI引数の定義・パース
import wandb  # 実験ログ・メトリクス記録
from omegaconf import OmegaConf  # YAML設定の読み込み
from calt.io import IOPipeline, UserCallableLoadPreprocessor  # データ・tokenizer構築
from calt.models import ModelPipeline  # モデル構築
from calt.trainer import TrainerPipeline  # 学習・評価パイプライン
from src.addition import format_problem  # 足し算問題の前処理
from src.addition_metrics import compute_metrics  # 評価指標（正答率など）


@click.command()
@click.option('--config_path', required=True, type=click.Path(exists=True),
              help='学習設定のYAMLファイル')
def main(config_path):
    # 2. データとtokenizerを用意
    cfg = OmegaConf.load(config_path)
    set_seed(cfg.train.seed)
    pipeline = IOPipeline.from_config(cfg.data)
    pipeline.dataset_load_preprocessor = UserCallableLoadPreprocessor(format_problem)
    io = pipeline.build()

    # 4. モデル生成
    model = ModelPipeline.from_io_dict(cfg.model, io).build()

    # 5. 学習・保存
    trainer = TrainerPipeline.from_io_dict(cfg.train, model, io).build()
    trainer.trainer.compute_metrics = compute_metrics
    trainer.train()

    trainer.save_model(f'{cfg.train.output_dir}/model')

    # 6. 評価
    accuracy = trainer.evaluate_and_save_generation(max_length=8)
    wandb.log({'accuracy': accuracy})
    print(accuracy)
    wandb.finish()


if __name__ == '__main__':
    main()
