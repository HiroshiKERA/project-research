"""発展課題D：token IDからベクトルを作る。"""
import torch
from torch import nn
from calt.models import register_input_embedding


class NumberEmbedding(nn.Module):
    """数字の値に比例したベクトルを、通常のtoken埋め込みに加える。"""
    def __init__(self, vocab_size, d_model, digit_ids, **kwargs):
        """埋め込みのパラメータを準備する。

        Args:
            vocab_size (int): 特殊tokenを含む語彙全体のtoken数。
            d_model (int): 1つのtokenを表すベクトルの次元数。例: 128。
            digit_ids (list[int]): 文字"0", "1", ..., "9"に対応する
                token IDを、この順で並べた長さ10のリスト。
                数字の値そのものとは異なる（"0"のIDが0とは限らない）。
            **kwargs: 共通の呼び出し側から渡される追加の名前付き引数。
                このクラスでは使わない。
        """
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.value_direction = nn.Parameter(torch.zeros(d_model))
        values = torch.zeros(vocab_size)
        values[digit_ids] = torch.arange(10, dtype=torch.float32) / 9
        self.register_buffer("values", values)

    def forward(self, input_ids):
        """token IDをベクトルに変換する。

        Args:
            input_ids (torch.LongTensor): 形状(B, L)の整数テンソル。
                Bは一度に処理する系列数、Lは各系列のtoken数。
                各要素は0以上vocab_size未満のtoken ID。
                数字・演算記号・特殊tokenのIDが含まれる。

        Returns:
            torch.Tensor: 形状(B, L, d_model)の浮動小数点テンソル。
                例: 入力が(2, 8)、d_model=128なら出力は(2, 8, 128)。
                数字には「数字の値/9 × 学習可能なベクトル」を加える。
                数字以外のtokenではこの加算項は0になる。
        """
        # EXERCISE_BEGIN: D1_NUMBER_EMBEDDING
        raise NotImplementedError("ここにコードを書く")
        # EXERCISE_END: D1_NUMBER_EMBEDDING


class AbacusEmbedding(nn.Module):
    """数字列ごとの桁位置を、token埋め込みに加える。"""
    def __init__(self, vocab_size, d_model, digit_ids,
                 capacity=256, max_offset=100, **kwargs):
        """tokenと数字列内の位置の埋め込みを準備する。

        Args:
            vocab_size (int): 特殊tokenを含む語彙全体のtoken数。
            d_model (int): 各埋め込みベクトルの次元数。
            digit_ids (list[int]): "0"〜"9"のtoken IDを数字順に並べた
                長さ10のリスト。数字tokenを判別するために使う。
            capacity (int): 位置埋め込み表の行数。既定値256。
                使用する位置IDは0以上capacity未満である必要がある。
                最長の数字列の長さ + max_offset - 1より大きくする。
            max_offset (int): 学習時に位置へ加える乱数の範囲。1以上。
                0〜max_offset-1から選ぶ。既定値100なら0〜99。
                評価時には加えない。
            **kwargs: 共通の呼び出し側から渡される追加の名前付き引数。
                このクラスでは使わない。
        """
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(capacity, d_model)
        self.register_buffer("digit_ids", torch.tensor(digit_ids))
        self.max_offset = max_offset

    def position_ids(self, input_ids):
        """数字が連続する区間ごとに、左からの位置IDを作る。

        Args:
            input_ids (torch.LongTensor): 形状(B, L)のtoken ID。
                Bは系列数、Lはtoken数。整数そのものの配列ではない。

        Returns:
            torch.LongTensor: 入力と同じ形状(B, L)・デバイスの位置ID。
                数字は各区間の先頭から1, 2, ...、数字以外は0。
                評価時、token列が["1", "2", "+", "8", "="]なら
                位置IDは[1, 2, 0, 1, 0]になる。
                学習時は全系列で共通のoffsetを選び、数字の位置だけに
                加える。例: offset=3なら[4, 5, 0, 4, 0]。

        model.train()では学習時、model.eval()では評価時の動作になる。
        位置は数の上位・下位ではなく、入力された文字列の左から数える。
        """
        # EXERCISE_BEGIN: D2_ABACUS_POSITIONS
        raise NotImplementedError("ここにコードを書く")
        # EXERCISE_END: D2_ABACUS_POSITIONS

    def forward(self, input_ids):
        """tokenの埋め込みに、数字列内の位置の埋め込みを加える。

        Args:
            input_ids (torch.LongTensor): 形状(B, L)のtoken ID。
                各要素は0以上vocab_size未満。Bは系列数、Lはtoken数。

        Returns:
            torch.Tensor: 形状(B, L, d_model)の浮動小数点テンソル。
                各tokenのベクトルにposition_idsで得た位置のベクトルを
                加えたもの。数字以外にも位置ID=0のベクトルを加える。
                位置ID=0のベクトルがゼロベクトルとは限らない。
        """
        # EXERCISE_BEGIN: D3_ABACUS_EMBEDDING
        raise NotImplementedError("ここにコードを書く")
        # EXERCISE_END: D3_ABACUS_EMBEDDING


# YAMLのinput_embedding_typeで選ぶ名前を登録する。
register_input_embedding("number", NumberEmbedding)
register_input_embedding("abacus", AbacusEmbedding)
