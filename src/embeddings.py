"""数値と桁位置を使う入力埋め込み。"""
import torch
from torch import nn
from calt.models import register_input_embedding


class NumberEmbedding(nn.Module):
    """数字の値に比例したベクトルを、通常のtoken埋め込みに加える。"""
    def __init__(self, vocab_size, d_model, digit_ids, **kwargs):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.value_direction = nn.Parameter(torch.zeros(d_model))
        # digit_idsは文字'0'〜'9'のtoken ID。数字の値とは異なる。
        values = torch.zeros(vocab_size)
        values[digit_ids] = torch.arange(10, dtype=torch.float32) / 9
        self.register_buffer('values', values)

    def forward(self, input_ids):
        """(B, L)のtoken IDを(B, L, D)のベクトルに変換する。"""
        # EXERCISE_BEGIN: D1_NUMBER_EMBEDDING
        token_vectors = self.token(input_ids)
        numbers = self.values[input_ids].unsqueeze(-1)
        return token_vectors + numbers * self.value_direction
        # EXERCISE_END: D1_NUMBER_EMBEDDING


class AbacusEmbedding(nn.Module):
    """数字列ごとの桁位置を、token埋め込みに加える。"""
    def __init__(self, vocab_size, d_model, digit_ids,
                 capacity=256, max_offset=100, **kwargs):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(capacity, d_model)
        self.register_buffer('digit_ids', torch.tensor(digit_ids))
        self.max_offset = max_offset

    def position_ids(self, input_ids):
        """数字列内で1, 2, …と数える。数字以外は0。戻り値は(B, L)。"""
        # EXERCISE_BEGIN: D2_ABACUS_POSITIONS
        is_digit = torch.isin(input_ids, self.digit_ids)
        positions = torch.zeros_like(input_ids)
        count = torch.zeros_like(input_ids[:, 0])
        for i in range(input_ids.shape[1]):
            count = (count + 1) * is_digit[:, i]
            positions[:, i] = count

        if self.training:
            # 同じバッチ内の数字には共通のoffsetを加える。
            offset = torch.randint(self.max_offset, (), device=input_ids.device)
            positions = positions + offset * is_digit
        return positions
        # EXERCISE_END: D2_ABACUS_POSITIONS

    def forward(self, input_ids):
        """数字以外には位置ID=0の学習可能なベクトルを加える。"""
        # EXERCISE_BEGIN: D3_ABACUS_EMBEDDING
        positions = self.position_ids(input_ids)
        return self.token(input_ids) + self.position(positions)
        # EXERCISE_END: D3_ABACUS_EMBEDDING


register_input_embedding('number', NumberEmbedding)
register_input_embedding('abacus', AbacusEmbedding)
