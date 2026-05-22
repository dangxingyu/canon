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


def _resolve_text_config(cfg):
    """Return the text-side config for both flat and nested (multimodal) configs.

    Gemma-3n / Gemma-4 store the language-model fields under ``cfg.text_config``;
    Qwen / Llama-style configs put them directly on ``cfg``.
    """
    return getattr(cfg, "text_config", cfg)


def summarize_config(model_path: str) -> ArchitectureSummary:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_cfg = _resolve_text_config(cfg)
    num_heads = int(text_cfg.num_attention_heads)
    num_kv_heads = int(text_cfg.num_key_value_heads)
    raw_int = getattr(text_cfg, "intermediate_size", None)
    if isinstance(raw_int, (list, tuple)):
        intermediate_size = int(max(raw_int))  # report the wider slab; layers may vary
    else:
        intermediate_size = int(raw_int or 0)
    head_dim_attr = getattr(text_cfg, "head_dim", None)
    head_dim = int(head_dim_attr) if head_dim_attr is not None else int(text_cfg.hidden_size // num_heads)
    q_per_kv = num_heads // num_kv_heads

    canonicalizers: list[str] = []
    notes: list[str] = []

    if num_heads % num_kv_heads == 0:
        canonicalizers.append("gqa_value_output_per_kv_head")
    else:
        notes.append("Q heads are not divisible by KV heads; GQA V/O grouping needs custom handling.")

    hidden_act = getattr(text_cfg, "hidden_act", None) or getattr(text_cfg, "hidden_activation", None)
    if hidden_act in {"silu", "gelu_pytorch_tanh", "gelu", "gelu_new"}:
        canonicalizers.append("swiglu_up_down_positive_channel_scaling")
    else:
        notes.append(f"MLP activation is {hidden_act!r}; verify gated structure before MLP canon.")

    # Detect head-dim normalization layers (Gemma-3n / Gemma-4) that break SPD V/O canon.
    if str(getattr(cfg, "model_type", "")).startswith(("gemma3n", "gemma4")):
        notes.append(
            "v_norm / q_norm / k_norm present on attention head_dim; SPD V/O canon is NOT "
            "function-preserving here -- use orthogonal V/O canon (vo_orth)."
        )

    if int(getattr(text_cfg, "num_kv_shared_layers", 0)) > 0:
        notes.append(
            f"num_kv_shared_layers={text_cfg.num_kv_shared_layers}; the last "
            f"{text_cfg.num_kv_shared_layers} layers reuse K/V from earlier source "
            "layers -- canon should skip them via is_kv_shared_layer."
        )

    if bool(getattr(text_cfg, "tie_word_embeddings", False) or getattr(cfg, "tie_word_embeddings", False)):
        notes.append("LM head appears tied; do not apply untied lm-head row-centering.")

    arch = getattr(cfg, "architectures", ["unknown"])[0]
    return ArchitectureSummary(
        model_path=str(Path(model_path)),
        model_type=str(cfg.model_type),
        architecture=str(arch),
        num_hidden_layers=int(text_cfg.num_hidden_layers),
        hidden_size=int(text_cfg.hidden_size),
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        query_heads_per_kv=q_per_kv,
        tie_word_embeddings=bool(
            getattr(text_cfg, "tie_word_embeddings", False) or getattr(cfg, "tie_word_embeddings", False)
        ),
        vocab_size=int(getattr(text_cfg, "vocab_size", getattr(cfg, "vocab_size", 0))),
        max_position_embeddings=int(getattr(text_cfg, "max_position_embeddings", 0)),
        canonicalizers=canonicalizers,
        notes=notes,
    )


def get_decoder_layers(model):
    """Locate the language-model decoder layer list.

    Supports:
      - Llama / Qwen / Gemma1/2/3 dense:          model.model.layers
      - GPT-2-style:                              model.transformer.h
      - Gemma-3n / Gemma-4 multimodal CausalLM:   model.model.language_model.layers
      - Gemma-3n / Gemma-4 ConditionalGeneration: model.language_model.model.layers
    """
    # Most common dense path.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # Gemma multimodal: language_model is nested either under .model or at the top level.
    candidates = []
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        candidates.append(model.model.language_model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    for lm in candidates:
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
    raise AttributeError(
        "Could not find decoder layers at model.model.layers, "
        "model.transformer.h, model.(model.)language_model(.model).layers"
    )


def get_text_config(model):
    """Return the language-model side config for both flat and nested architectures."""
    return _resolve_text_config(model.config)

