"""保存モデルの桁数別評価と描画を実行する。"""
import click
from omegaconf import OmegaConf
from support.length_evaluation import evaluate


@click.command()
@click.option('--config_path', required=True, type=click.Path(exists=True),
              help='評価設定のYAMLファイル')
def main(config_path):
    cfg = OmegaConf.load(config_path)
    evaluate(cfg)


if __name__ == '__main__':
    main()
