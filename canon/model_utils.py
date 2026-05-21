from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchitectureSummary:
    model_path: str
    model_type: str
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    query_heads_per_kv: int
    tie_word_embeddings: bool
    vocab_size: int
    max_position_embeddings: int
    canonicalizers: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_config(model_path: str) -> ArchitectureSummary:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    q_per_kv = num_heads // num_kv_heads

    canonicalizers: list[str] = []
    notes: list[str] = []

    if num_heads % num_kv_heads == 0:
        canonicalizers.append("gqa_value_output_per_kv_head")
    else:
        notes.append("Q heads are not divisible by KV heads; GQA V/O grouping needs custom handling.")

    if getattr(cfg, "hidden_act", None) == "silu":
        canonicalizers.append("swiglu_up_down_positive_channel_scaling")
    else:
        notes.append(f"MLP activation is {getattr(cfg, 'hidden_act', None)!r}; verify SwiGLU structure before scaling.")

    if bool(getattr(cfg, "tie_word_embeddings", False)):
        notes.append("LM head appears tied; do not apply untied lm-head row-centering.")

    arch = getattr(cfg, "architectures", ["unknown"])[0]
    return ArchitectureSummary(
        model_path=str(Path(model_path)),
        model_type=str(cfg.model_type),
        architecture=str(arch),
        num_hidden_layers=int(cfg.num_hidden_layers),
        hidden_size=int(cfg.hidden_size),
        intermediate_size=int(cfg.intermediate_size),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        query_heads_per_kv=q_per_kv,
        tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        vocab_size=int(cfg.vocab_size),
        max_position_embeddings=int(getattr(cfg, "max_position_embeddings", 0)),
        canonicalizers=canonicalizers,
        notes=notes,
    )


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not find decoder layers at model.model.layers or model.transformer.h")

