"""発展課題の学習設定から保存モデルを選び、桁数別に評価する。"""
import click
from omegaconf import OmegaConf
from support.length_evaluation import evaluate


@click.command()
@click.option('--config_path', required=True, type=click.Path(exists=True))
@click.option('--evaluation_config', default='configs/extensions/evaluate.yaml',
              type=click.Path(exists=True))
def main(config_path, evaluation_config):
    train_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.load(evaluation_config)
    # モデルごとの保存先と学習時のtoken分割を引き継ぐ。
    cfg.model_path = f'{train_cfg.train.output_dir}/model'
    cfg.lexer_config = train_cfg.data.lexer_config
    cfg.output_dir = f'{train_cfg.train.output_dir}/evaluation'
    evaluate(cfg)


if __name__ == '__main__':
    main()
