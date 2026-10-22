"""保存したCALT decoder-onlyモデルを桁数別に評価する。"""
import csv
import importlib
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from calt.io import IOPipeline
from calt.io.base import StandardDataCollator
from calt.models import get_input_embedding
from calt.models.decoder_only import DecoderOnlyConfig, DecoderOnlyTransformer
from omegaconf import OmegaConf
from safetensors.torch import load_model
from transformers import PreTrainedTokenizerFast


def load_function(name):
    """例：src.addition:format_problem、operator:add。"""
    module, function = name.split(':')
    return getattr(importlib.import_module(module), function)


def make_problems(digits, samples_per_cell, seed, calculate):
    """桁数の組ごとに異なる数値ペアを生成する。1桁には0を含む。"""
    for m in digits:
        for n in digits:
            low_a = 0 if m == 1 else 10 ** (m - 1)
            low_b = 0 if n == 1 else 10 ** (n - 1)
            size_a, size_b = 10**m - low_a, 10**n - low_b
            count = min(samples_per_cell, size_a * size_b)
            rng = random.Random(f'{seed}:{m}:{n}')
            chosen = set()
            while len(chosen) < count:
                a = rng.randrange(low_a, 10**m)
                b = rng.randrange(low_b, 10**n)
                if (a, b) in chosen:
                    continue
                chosen.add((a, b))
                yield {'a': a, 'b': b, 'answer': calculate(a, b),
                       'digits_a': m, 'digits_b': n}


def restore_model(model_path, tokenizer):
    """checkpointの構造と重みを復元する。埋め込みの定義はsrc.embeddingsを使う。"""
    config = DecoderOnlyConfig.from_pretrained(model_path)
    if config.model_type != 'decoder_only':
        raise ValueError('この評価コードはCALTのdecoder-onlyモデル用です。')
    model = DecoderOnlyTransformer(config)
    kind = getattr(config, 'input_embedding_type', 'token')
    if kind != 'token':
        importlib.import_module('src.embeddings')
        options = dict(getattr(config, 'input_embedding_kwargs', {}))
        options.setdefault('digit_ids', [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)])
        model.embedding = get_input_embedding(kind, config.vocab_size, config.d_model, **options)
    load_model(model, str(Path(model_path) / 'model.safetensors'))
    return model


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(cfg):
    """表記を変換してバッチ生成し、問題ごとの結果と桁数を保存する。"""
    digits = list(cfg.digits)
    if not digits or len(set(digits)) != len(digits) or any(d < 1 for d in digits):
        raise ValueError('digitsには重複のない正の桁数を指定してください。')
    if cfg.samples_per_cell < 1 or cfg.batch_size < 1 or cfg.max_length < 1:
        raise ValueError('件数・バッチサイズ・生成長は正の整数を指定してください。')
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'evaluate.yaml')
    format_problem = load_function(cfg.format_function)
    calculate = load_function(cfg.answer_function)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.model_path)
    # 保存したtokenizerを使い、学習時と同じlexerで文字列を桁に分割する。
    lexer = IOPipeline.from_config(OmegaConf.create({'lexer_config': cfg.lexer_config})).preprocessor
    collator = StandardDataCollator(tokenizer)
    model = restore_model(cfg.model_path, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    problems = list(make_problems(digits, cfg.samples_per_cell, cfg.seed, calculate))
    results = []
    for start in range(0, len(problems), cfg.batch_size):
        examples = problems[start:start + cfg.batch_size]
        formatted = []
        for row in examples:
            problem, answer = format_problem([row['a'], row['b']], row['answer'])
            formatted.append({'input': lexer.encode(problem), 'target': lexer.encode(answer)})
        batch = collator(formatted)
        if any((tensor == tokenizer.unk_token_id).any().item()
               for tensor in (batch['input_ids'], batch['labels'])):
            raise ValueError('評価データに学習時の語彙にないtokenがあります。lexerと表記を確認してください。')
        with torch.inference_mode():
            generated = model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                max_length=cfg.max_length,
            ).cpu().tolist()
        for row, form, prediction, labels in zip(examples, formatted, generated, batch['labels'].tolist()):
            target = [t for t in labels if t != -100]
            if prediction and prediction[0] == tokenizer.bos_token_id:
                prediction = prediction[1:]
            ended = tokenizer.eos_token_id in prediction
            if ended:
                prediction = prediction[:prediction.index(tokenizer.eos_token_id) + 1]
            results.append({**row, 'input': form['input'],
                            'target': tokenizer.decode(target, skip_special_tokens=True),
                            'prediction': tokenizer.decode(prediction, skip_special_tokens=True),
                            'eos': ended, 'exact': prediction == target})
        print(f'{min(start + cfg.batch_size, len(problems))}/{len(problems)}', flush=True)
    path = output / 'predictions.csv'
    write_csv(path, results)
    plot_results(path)
    return results


def plot_results(predictions_path):
    """CSVからヒートマップとn桁同士の曲線を描く。モデルの再実行は不要。"""
    predictions_path = Path(predictions_path)
    with predictions_path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('評価結果が空です。')
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['digits_a']), int(row['digits_b'])].append(row['exact'] == 'True')
    summary = [{'digits_a': m, 'digits_b': n, 'count': len(values),
                'exact_match': sum(values) / len(values)}
               for (m, n), values in sorted(groups.items())]
    output = predictions_path.parent
    write_csv(output / 'accuracy_by_digits.csv', summary)
    a_digits = sorted({m for m, n in groups})
    b_digits = sorted({n for m, n in groups})
    matrix = np.full((len(a_digits), len(b_digits)), np.nan)
    for i, m in enumerate(a_digits):
        for j, n in enumerate(b_digits):
            if (m, n) in groups:
                matrix[i, j] = np.mean(groups[m, n])
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    image = ax.imshow(matrix, origin='lower', aspect='auto', vmin=0, vmax=1, cmap=cmap)
    for values, setter in [(a_digits, ax.set_yticks), (b_digits, ax.set_xticks)]:
        indexes = list(range(0, len(values), max(1, (len(values) + 14) // 15)))
        if indexes[-1] != len(values) - 1:
            indexes.append(len(values) - 1)
        setter(indexes, [values[i] for i in indexes])
    ax.set(xlabel='Digits of operand B (n)', ylabel='Digits of operand A (m)',
           title='Exact match by operand length')
    fig.colorbar(image, ax=ax, label='Exact match rate')
    fig.tight_layout()
    fig.savefig(output / 'length_heatmap.png', dpi=180)
    plt.close(fig)

    diagonal = [r for r in summary if r['digits_a'] == r['digits_b']]
    if diagonal:
        write_csv(output / 'accuracy_same_digits.csv', diagonal)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r['digits_a'] for r in diagonal], [r['exact_match'] for r in diagonal], 'o-')
        ax.set(xlabel='Digits per operand (n)', ylabel='Exact match rate', ylim=(-0.02, 1.02),
               title='Exact match for two n-digit operands')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / 'length_curve.png', dpi=180)
        plt.close(fig)
    return summary
