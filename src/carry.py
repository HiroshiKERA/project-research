"""各桁から次の桁への繰り上がりを補助予測する。"""
import torch
from calt.io.base import StandardDataCollator


def carry_targets(left, right):
    """下位から並んだ同じ桁数の数字列を受け取る。例: '95', '10'。"""
    # EXERCISE_BEGIN: CARRY_TARGETS
    carry = 0
    targets = []
    for a, b in zip(left, right):
        carry = int(int(a) + int(b) + carry >= 10)
        targets.append(carry)
    return targets
    # EXERCISE_END: CARRY_TARGETS


class CarryCollator(StandardDataCollator):
    """src.extension_tasks.format_problemの逆順・0埋め表記専用。"""
    def __call__(self, examples):
        batch = super().__call__(examples)
        carry_labels = torch.full_like(batch['labels'], -100)
        for row, example in enumerate(examples):
            text = example['input'].replace(' ', '').removesuffix('=')
            left, right = text.split('+')
            targets = carry_targets(left, right)
            # labelsの先頭は一の位。最上位の追加桁・EOS・PADは対象外。
            carry_labels[row, :len(targets)] = torch.tensor(targets)
        batch['carry_labels'] = carry_labels
        return batch
