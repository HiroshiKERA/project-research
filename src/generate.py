"""足し算の学習・評価用データを生成する。"""
import click
from omegaconf import OmegaConf
from calt.dataset import DatasetPipeline
from src.addition import sum_problem


@click.command()
@click.option('--config_path', required=True, type=click.Path(exists=True),
              help='データ生成設定のYAMLファイル')
def main(config_path):
    # 1. 問題を生成
    cfg = OmegaConf.load(config_path)
    DatasetPipeline.from_config(cfg, instance_generator=sum_problem).run()


if __name__ == '__main__':
    main()
