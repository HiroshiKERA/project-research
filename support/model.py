"""CALT 1.5.0のdecoder-onlyへ入力埋め込みregistryを接続する。"""
from calt.models import get_input_embedding
from omegaconf import OmegaConf
from src import embeddings  # 学習・再読込の両方で登録する


def connect_embeddings(model, cfg, tokenizer):
    kind = cfg.get("input_embedding_type", "token")
    if kind == "token":
        return model
    options = OmegaConf.to_container(cfg.get("input_embedding_kwargs", OmegaConf.create({})), resolve=True)
    digit_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]
    embedding = get_input_embedding(kind, model.config.vocab_size,
                                    model.config.d_model, digit_ids=digit_ids, **options)
    embedding.apply(model._init_weights)
    model.embedding = embedding
    model.config.input_embedding_type = kind
    model.config.input_embedding_kwargs = dict(options, digit_ids=digit_ids)
    return model
