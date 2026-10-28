"""保存済みの評価CSVから手法ごとの完全一致率を比較する。"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf


def plot_comparison(config_paths, output_dir='runs/extensions-comparison'):
    """config_paths: 表示名と学習設定YAMLの辞書。"""
    records = {}
    reference = None
    for name, path in config_paths.items():
        train_cfg = OmegaConf.load(path)
        folder = Path(train_cfg.train.output_dir) / 'evaluation'
        with (folder / 'predictions.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        problems = [(r['a'], r['b'], r['answer'], r['digits_a'], r['digits_b']) for r in rows]
        if reference is None:
            reference = problems
        elif problems != reference:
            raise ValueError('比較する評価問題が異なります。同じ評価設定で再評価してください。')
        with (folder / 'accuracy_same_digits.csv').open(newline='') as f:
            records[name] = list(csv.DictReader(f))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in records.items():
        ax.plot([int(r['digits_a']) for r in rows],
                [float(r['exact_match']) for r in rows], 'o-', label=name)
    ax.set(xlabel='Digits per operand (n)', ylabel='Exact match rate', ylim=(-0.02, 1.02))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output / 'length_comparison.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
