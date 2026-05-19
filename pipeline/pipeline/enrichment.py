"""Data model for enriched model specifications."""

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Validated model specification."""

    model_name: str
    learnable_params_b: float | None = Field(
        None, description="True logical param count in billions (from config-based counting)"
    )
    active_params_b: float | None = Field(None, description="Active params for MoE models")
    architecture: str | None = Field("Dense", description="Dense or MoE")
    context_length: int | None = None
    precision: str | None = Field(
        None, description="Published precision, e.g. FP8, BF16, INT4-mixed"
    )
    attention_type: str | None = Field(None, description="'MLA', 'GQA', or 'DSV4'")
    num_hidden_layers: int | None = Field(None, description="Number of transformer layers")
    num_kv_layers: int | None = Field(None, description="Layers with KV cache (None = all layers)")
    num_kv_heads: int | None = Field(None, description="Number of KV heads (GQA only)")
    head_dim: int | None = Field(None, description="Head dimension (GQA only)")
    routed_expert_params_b: float | None = Field(
        None, description="Routed expert params in billions (the INT4-quantized bulk)"
    )
    kv_lora_rank: int | None = Field(None, description="KV LoRA rank (MLA only)")
    qk_rope_head_dim: int | None = Field(None, description="RoPE head dimension (MLA only)")
    kv_elems_per_token: int | None = Field(
        None,
        description=(
            "Precomputed per-token KV-cache element width summed across all "
            "KV-bearing layers (compressed-attention archs, e.g. deepseek_v4 "
            "'DSV4'). The web multiplies this by the KV precision bytes."
        ),
    )
    hf_model_id: str | None = Field(None, description="HuggingFace repo ID for linking")
    model_url: str | None = Field(
        None, description="Fallback URL (GitHub, blog, etc.) for models with no HF page"
    )
    license_name: str | None = Field(
        None, description="Human-readable license name (e.g., 'MIT', 'Apache 2.0')"
    )
    license_url: str | None = Field(None, description="URL to the license text")
