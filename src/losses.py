"""正解tokenの確率から学習用のlossを計算する。"""
import torch
import torch.nn.functional as F


def token_losses(logits, labels):
    """logits: (B, L, V)、labels: (B, L)。paddingの正解は-100。"""
    return F.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction='none',
    )


def mean_token_loss(logits, labels):
    losses = token_losses(logits, labels)
    return losses.sum() / (labels != -100).sum()


def sequence_loss(logits, labels, strength=1.0, power=2.0):
    """正解系列の確率qに対して、通常のCE + strength * (1 - q**power)。"""
    # EXERCISE_BEGIN: SEQUENCE_LOSS
    losses = token_losses(logits, labels)
    ce = losses.sum() / (labels != -100).sum()
    log_q = -losses.sum(dim=1)  # EOSを含む正解系列の対数確率
    reward = torch.exp(power * log_q)
    return ce + strength * (1 - reward).mean()
    # EXERCISE_END: SEQUENCE_LOSS
