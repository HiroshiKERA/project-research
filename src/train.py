"""足し算のモデルを学習・評価する。"""
import click
import wandb
from omegaconf import OmegaConf
from calt.io import IOPipeline, UserCallableLoadPreprocessor
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline
from src.addition import format_problem
from src.addition_metrics import compute_metrics


@click.command()
@click.option('--config_path', required=True, type=click.Path(exists=True),
              help='学習設定のYAMLファイル')
def main(config_path):
    # 2. データとtokenizerを用意
    cfg = OmegaConf.load(config_path)
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
