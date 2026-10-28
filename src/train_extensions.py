"""発展課題：YAMLで選んだ変更を足し算モデルに適用する。"""
from pathlib import Path
from functools import partial
import click
import wandb
from omegaconf import OmegaConf
from calt.io import IOPipeline, UserCallableLoadPreprocessor
from calt.models import ModelPipeline
from calt.trainer import TrainerPipeline
from transformers import set_seed
from src.extension_tasks import format_problem
from src.extension_models import CarryDecoder, SequenceLossDecoder
from src.carry import CarryCollator
from src.addition_metrics import compute_metrics
from support.model import connect_embeddings


@click.command()
@click.option('--config_path', required=True, type=click.Path(exists=True))
def main(config_path):
    cfg = OmegaConf.load(config_path)
    set_seed(cfg.train.seed)
    pipeline = IOPipeline.from_config(cfg.data)
    pipeline.dataset_load_preprocessor = UserCallableLoadPreprocessor(format_problem)
    io = pipeline.build()
    model = ModelPipeline.from_io_dict(cfg.model, io).build()

    # 補助予測・lossの実験では、同じ初期重みを継承モデルへ渡す。
    if cfg.experiment in ('carry', 'sequence'):
        base = model
        if cfg.experiment == 'carry':
            base.config.carry_weight = cfg.carry_weight
            model = CarryDecoder(base.config)
            io['data_collator'] = CarryCollator(io['tokenizer'])
        else:
            base.config.sequence_weight = cfg.sequence_weight
            base.config.sequence_power = cfg.sequence_power
            model = SequenceLossDecoder(base.config)
        model.load_state_dict(base.state_dict(), strict=False)
    model.config.extension_experiment = cfg.experiment
    connect_embeddings(model, cfg.model, io['tokenizer'])

    trainer = TrainerPipeline.from_io_dict(cfg.train, model, io).build()
    trainer.trainer.compute_metrics = compute_metrics
    # 学習中の生成評価にも、YAMLで指定した生成長を使う。
    trainer.trainer.evaluate_and_save_generation = partial(
        trainer.trainer.evaluate_and_save_generation,
        max_length=cfg.generation_max_length,
    )
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.train.output_dir) / 'train.yaml')
    set_seed(cfg.train.seed)
    trainer.train()
    trainer.save_model(f'{cfg.train.output_dir}/model')
    accuracy = trainer.evaluate_and_save_generation(max_length=cfg.generation_max_length)
    wandb.log({'accuracy': accuracy})
    wandb.finish()


if __name__ == '__main__':
    main()
