"""CALTの系列処理を再利用し、隠れ状態とlogitsをloss計算へ渡す。"""
import torch
from calt.models.decoder_only import DecoderOnlyTransformer
from transformers.modeling_outputs import CausalLMOutput
from src.losses import mean_token_loss


class LossEditableDecoder(DecoderOnlyTransformer):
    """生成処理はCALTを継承。変更するのはtraining_loss。"""

    def training_loss(self, hidden, logits, labels, carry_labels=None):
        return mean_token_loss(logits, labels)

    def forward(self, input_ids=None, attention_mask=None,
                decoder_input_ids=None, decoder_attention_mask=None,
                labels=None, carry_labels=None, return_dict=None, **kwargs):
        packed, padding_mask, starts = self._pack(
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,
        )
        hidden = self._run_stack(packed, padding_mask)
        offsets = torch.arange(decoder_input_ids.size(1), device=hidden.device)
        indices = (starts[:, None] + offsets).clamp(max=packed.size(1) - 1)
        target_hidden = hidden.gather(
            1, indices[:, :, None].expand(-1, -1, hidden.size(-1)),
        )
        logits = self.lm_head(target_hidden)
        loss = None if labels is None else self.training_loss(
            target_hidden, logits, labels, carry_labels,
        )
        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)
        # CALTと同じく、compute_metricsにはtoken IDを返す。
        return CausalLMOutput(loss=loss, logits=logits.argmax(dim=-1))
