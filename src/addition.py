"""足し算の問題生成と、学習に使う表記への変換。"""
import random


def sum_problem(seed):
    """各整数の桁数を1〜5から一様に選び、その和を返す。"""
    rng = random.Random(seed)
    da, db = rng.randint(1, 5), rng.randint(1, 5)
    a = rng.randint(0 if da == 1 else 10 ** (da - 1), 10 ** da - 1)
    b = rng.randint(0 if db == 1 else 10 ** (db - 1), 10 ** db - 1)
    return [a, b], a + b


def format_problem(problem, answer):
    """問題のリストと正解の整数を文字列に変換する。"""
    return f"{problem[0]}+{problem[1]}=", f"{answer}"
