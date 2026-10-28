"""補助予測とlossの変更。Transformerと生成処理は継承する。"""
from torch import nn
import torch.nn.functional as F
from support.extension_model import LossEditableDecoder
from src.losses import mean_token_loss, sequence_loss


class CarryDecoder(LossEditableDecoder):
    def __init__(self, config):
        super().__init__(config)
        # EXERCISE_BEGIN: CARRY_HEAD
        self.carry_head = nn.Linear(config.d_model, 1)
        self.carry_head.apply(self._init_weights)
        # EXERCISE_END: CARRY_HEAD

    def training_loss(self, hidden, logits, labels, carry_labels=None):
        # hidden: (B, L, D)。各桁を予測する時点の隠れ状態。
        # EXERCISE_BEGIN: CARRY_LOSS
        main_loss = mean_token_loss(logits, labels)
        if carry_labels is None:
            return main_loss
        carry_logits = self.carry_head(hidden).squeeze(-1)
        mask = carry_labels != -100
        aux_loss = F.binary_cross_entropy_with_logits(
            carry_logits[mask], carry_labels[mask].float(),
        )
        return main_loss + self.config.carry_weight * aux_loss
        # EXERCISE_END: CARRY_LOSS


class SequenceLossDecoder(LossEditableDecoder):
    def training_loss(self, hidden, logits, labels, carry_labels=None):
        return sequence_loss(logits, labels,
                             strength=self.config.sequence_weight,
                             power=self.config.sequence_power)
