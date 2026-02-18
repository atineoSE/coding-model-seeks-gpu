"""Data model for enriched model specifications."""

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Validated model specification."""

    model_name: str
    published_param_count_b: float | None = Field(
        None, description="HF safetensors element count in billions (None when unavailable)"
    )
    learnable_params_b: float | None = Field(
        None, description="True logical param count in billions (from config-based counting)"
    )
    active_params_b: float | None = Field(None, description="Active params for MoE models")
    architecture: str | None = Field("Dense", description="Dense or MoE")
    context_length: int | None = None
    precision: str | None = Field(
        None, description="Published precision, e.g. FP8, BF16, INT4-mixed"
    )
    attention_type: str | None = Field(None, description="'MLA' or 'GQA'")
    num_hidden_layers: int | None = Field(None, description="Number of transformer layers")
    num_kv_heads: int | None = Field(None, description="Number of KV heads (GQA only)")
    head_dim: int | None = Field(None, description="Head dimension (GQA only)")
    routed_expert_params_b: float | None = Field(
        None, description="Routed expert params in billions (the INT4-quantized bulk)"
    )
    kv_lora_rank: int | None = Field(None, description="KV LoRA rank (MLA only)")
    qk_rope_head_dim: int | None = Field(None, description="RoPE head dimension (MLA only)")
    hf_model_id: str | None = Field(None, description="HuggingFace repo ID for linking")
