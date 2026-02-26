"""Config-based parameter counting and precision detection for known MoE architectures.

Pure computation — no I/O.  Raises ``ValueError`` on unknown architectures or
missing fields so the pipeline fails loudly rather than silently producing
wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


class AttentionType(Enum):
    MLA = "mla"
    GQA = "gqa"


@dataclass(frozen=True)
class MoEFieldMapping:
    """Maps architecture-specific config keys to a common vocabulary."""

    expert_count_key: str
    shared_expert_key: str
    shared_expert_is_count: bool  # True → count of experts; False → intermediate size (0=none)
    expert_intermediate_key: str
    dense_layers_key: str | None
    mtp_key: str | None
    num_mlp_projections: int = 3  # 3 = gated SwiGLU (gate+up+down), 2 = ungated (up+down)


@dataclass(frozen=True)
class ParamCountResult:
    total_params: int
    active_params: int
    routed_expert_params: int  # params in routed experts only (the quantizable bulk)
    model_type: str
    num_layers: int
    num_moe_layers: int
    num_dense_layers: int
    num_mamba_layers: int = 0
    num_attention_layers: int = 0  # 0 = all layers have attention (non-hybrid)


@dataclass(frozen=True)
class PrecisionInfo:
    bytes_per_param: float
    dtype_str: str
    is_mixed: bool = False


# ---------------------------------------------------------------------------
# Multimodal wrapper handling
# ---------------------------------------------------------------------------

MULTIMODAL_MODEL_TYPES = {"kimi_k25"}


def resolve_text_config(config: dict) -> dict:
    """Unwrap multimodal configs (e.g. Kimi-K2.5) to get the text backbone."""
    model_type = config.get("model_type", "")
    if model_type in MULTIMODAL_MODEL_TYPES:
        text_config = config.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError(
                f"model_type='{model_type}' requires 'text_config' dict, "
                f"but it is {'missing' if text_config is None else type(text_config).__name__}"
            )
        return text_config
    return config


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

KNOWN_ARCHITECTURES: dict[str, tuple[AttentionType, MoEFieldMapping]] = {
    "deepseek_v32": (
        AttentionType.MLA,
        MoEFieldMapping(
            expert_count_key="n_routed_experts",
            shared_expert_key="n_shared_experts",
            shared_expert_is_count=True,
            expert_intermediate_key="moe_intermediate_size",
            dense_layers_key="first_k_dense_replace",
            mtp_key="num_nextn_predict_layers",
        ),
    ),
    "kimi_k2": (
        AttentionType.MLA,
        MoEFieldMapping(
            expert_count_key="n_routed_experts",
            shared_expert_key="n_shared_experts",
            shared_expert_is_count=True,
            expert_intermediate_key="moe_intermediate_size",
            dense_layers_key="first_k_dense_replace",
            mtp_key="num_nextn_predict_layers",
        ),
    ),
    "glm4_moe": (
        AttentionType.GQA,
        MoEFieldMapping(
            expert_count_key="n_routed_experts",
            shared_expert_key="n_shared_experts",
            shared_expert_is_count=True,
            expert_intermediate_key="moe_intermediate_size",
            dense_layers_key="first_k_dense_replace",
            mtp_key="num_nextn_predict_layers",
        ),
    ),
    "qwen3_moe": (
        AttentionType.GQA,
        MoEFieldMapping(
            expert_count_key="num_experts",
            shared_expert_key="shared_expert_intermediate_size",
            shared_expert_is_count=False,
            expert_intermediate_key="moe_intermediate_size",
            dense_layers_key=None,
            mtp_key=None,
        ),
    ),
    "minimax_m2": (
        AttentionType.GQA,
        MoEFieldMapping(
            expert_count_key="num_local_experts",
            shared_expert_key="shared_intermediate_size",
            shared_expert_is_count=False,
            expert_intermediate_key="intermediate_size",
            dense_layers_key=None,
            mtp_key="num_mtp_modules",
        ),
    ),
    "nemotron_h": (
        AttentionType.GQA,
        MoEFieldMapping(
            expert_count_key="n_routed_experts",
            shared_expert_key="moe_shared_expert_intermediate_size",
            shared_expert_is_count=False,
            expert_intermediate_key="moe_intermediate_size",
            dense_layers_key=None,
            mtp_key=None,
            num_mlp_projections=2,
        ),
    ),
    "glm_moe_dsa": (
        AttentionType.MLA,
        MoEFieldMapping(
            expert_count_key="n_routed_experts",
            shared_expert_key="n_shared_experts",
            shared_expert_is_count=True,
            expert_intermediate_key="moe_intermediate_size",
            dense_layers_key="first_k_dense_replace",
            mtp_key="num_nextn_predict_layers",
        ),
    ),
    "qwen3_next": (
        AttentionType.GQA,
        MoEFieldMapping(
            expert_count_key="num_experts",
            shared_expert_key="shared_expert_intermediate_size",
            shared_expert_is_count=False,
            expert_intermediate_key="moe_intermediate_size",
            dense_layers_key=None,
            mtp_key=None,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(config: dict, key: str) -> int:
    """Extract a required integer config field."""
    val = config.get(key)
    if val is None:
        raise ValueError(f"Required config field '{key}' is missing")
    return int(val)


# ---------------------------------------------------------------------------
# Attention parameter functions (per layer)
# ---------------------------------------------------------------------------


def _mla_attention_params(config: dict) -> int:
    """MLA (Multi-head Latent Attention) params per layer."""
    hidden = _require(config, "hidden_size")
    n_heads = _require(config, "num_attention_heads")
    kv_lora_rank = _require(config, "kv_lora_rank")
    q_lora_rank = _require(config, "q_lora_rank")
    qk_nope = _require(config, "qk_nope_head_dim")
    qk_rope = _require(config, "qk_rope_head_dim")
    v_head = _require(config, "v_head_dim")

    q_a_proj = hidden * q_lora_rank
    q_a_norm = q_lora_rank
    q_b_proj = q_lora_rank * n_heads * (qk_nope + qk_rope)
    kv_a_proj = hidden * (kv_lora_rank + qk_rope)
    kv_a_norm = kv_lora_rank
    kv_b_proj = kv_lora_rank * n_heads * (qk_nope + v_head)
    o_proj = n_heads * v_head * hidden

    return q_a_proj + q_a_norm + q_b_proj + kv_a_proj + kv_a_norm + kv_b_proj + o_proj


def _gqa_attention_params(config: dict) -> int:
    """GQA (Grouped Query Attention) params per layer."""
    hidden = _require(config, "hidden_size")
    n_heads = _require(config, "num_attention_heads")
    n_kv_heads = _require(config, "num_key_value_heads")
    head_dim = config.get("head_dim", hidden // n_heads)

    q_proj = hidden * (n_heads * head_dim)
    k_proj = hidden * (n_kv_heads * head_dim)
    v_proj = hidden * (n_kv_heads * head_dim)
    o_proj = (n_heads * head_dim) * hidden

    return q_proj + k_proj + v_proj + o_proj


def _dsa_indexer_params(config: dict) -> int:
    """DSA (Dynamic Sparse Attention) indexer params per layer.

    Returns 0 if DSA fields are absent (non-DSA architectures).
    """
    index_n_heads = config.get("index_n_heads")
    index_head_dim = config.get("index_head_dim")
    if index_n_heads is None or index_head_dim is None:
        return 0

    hidden = _require(config, "hidden_size")
    index_n_heads = int(index_n_heads)
    index_head_dim = int(index_head_dim)

    # wq_b: maps from hidden → index_n_heads * index_head_dim
    wq_b = hidden * (index_n_heads * index_head_dim)
    # wk: maps from hidden → index_n_heads * index_head_dim
    wk = hidden * (index_n_heads * index_head_dim)
    # weights_proj: maps from index_n_heads * index_head_dim → index_n_heads
    weights_proj = (index_n_heads * index_head_dim) * index_n_heads
    # k_norm: LayerNorm with weight + bias over index_head_dim
    k_norm = 2 * index_head_dim

    return wq_b + wk + weights_proj + k_norm


def _qwen3_next_linear_attn_params(config: dict) -> int:
    """Qwen3-Next Mamba2-style linear attention params per layer."""
    hidden = _require(config, "hidden_size")
    n_attn_heads = _require(config, "num_attention_heads")
    head_dim = config.get("head_dim", hidden // n_attn_heads)

    linear_n_key_heads = _require(config, "linear_num_key_heads")
    linear_key_head_dim = _require(config, "linear_key_head_dim")
    linear_n_value_heads = _require(config, "linear_num_value_heads")
    linear_value_head_dim = _require(config, "linear_value_head_dim")
    conv_kernel = _require(config, "linear_conv_kernel_dim")

    q_dim = n_attn_heads * head_dim
    k_dim = linear_n_key_heads * linear_key_head_dim
    v_dim = linear_n_value_heads * linear_value_head_dim
    z_dim = n_attn_heads * head_dim  # gate, same as Q

    # in_proj_qkvz: combined Q/K/V/Z projection
    in_proj_qkvz = hidden * (q_dim + k_dim + v_dim + z_dim)
    # in_proj_ba: B/A state projections (B + A = 2 * n_key_heads * ssm_state)
    # From safetensors: shape is [hidden, 2 * linear_n_key_heads]
    in_proj_ba = hidden * (2 * linear_n_key_heads)
    # conv1d: depthwise causal convolution over (v_dim + k_dim)
    d_conv = v_dim + k_dim
    conv1d = d_conv * conv_kernel + d_conv  # weight + bias
    # out_proj: maps v_dim → hidden
    out_proj = v_dim * hidden
    # norm (RMSNorm over v_dim)
    norm = v_dim
    # A_log + dt_bias: per-head parameters
    a_log = n_attn_heads
    dt_bias = n_attn_heads

    return in_proj_qkvz + in_proj_ba + conv1d + out_proj + norm + a_log + dt_bias


# ---------------------------------------------------------------------------
# MLP / MoE parameter functions
# ---------------------------------------------------------------------------


def _dense_mlp_params(hidden: int, intermediate: int, *, num_projections: int = 3) -> int:
    """Dense MLP params.  Default 3 = gated SwiGLU (gate+up+down); 2 = ungated (up+down)."""
    return num_projections * hidden * intermediate


def _moe_layer_params(
    config: dict,
    mapping: MoEFieldMapping,
    num_active_experts: int | None = None,
) -> int:
    """MoE FFN params for one layer.

    If *num_active_experts* is given, counts only that many routed experts
    (for active-param calculation).  Otherwise counts all routed experts.
    """
    hidden = _require(config, "hidden_size")
    n_experts = _require(config, mapping.expert_count_key)
    expert_intermediate = _require(config, mapping.expert_intermediate_key)

    n_proj = mapping.num_mlp_projections

    # Routed experts
    active = num_active_experts if num_active_experts is not None else n_experts
    routed = active * _dense_mlp_params(hidden, expert_intermediate, num_projections=n_proj)

    # Router gate
    router = hidden * n_experts

    # Shared experts
    shared_val = _require(config, mapping.shared_expert_key)
    if mapping.shared_expert_is_count:
        # Value is count of shared experts, each using expert_intermediate
        shared = shared_val * _dense_mlp_params(hidden, expert_intermediate, num_projections=n_proj)
    else:
        # Value is intermediate size directly (0 = none)
        shared = (
            _dense_mlp_params(hidden, shared_val, num_projections=n_proj) if shared_val > 0 else 0
        )

    return routed + router + shared


# ---------------------------------------------------------------------------
# Mamba2 / hybrid architecture helpers
# ---------------------------------------------------------------------------


def _mamba2_layer_params(config: dict) -> int:
    """Mamba2 SSM params per layer."""
    hidden = _require(config, "hidden_size")
    num_heads = _require(config, "mamba_num_heads")
    head_dim = _require(config, "mamba_head_dim")
    ssm_state = _require(config, "ssm_state_size")
    n_group = config.get("n_group", 1)
    conv_kernel = _require(config, "conv_kernel")

    d_inner = num_heads * head_dim

    # in_proj: hidden → (2*d_inner + 2*n_group*ssm_state + num_heads)
    in_proj = hidden * (2 * d_inner + 2 * n_group * ssm_state + num_heads)
    # conv1d (depthwise): d_inner * conv_kernel + bias
    conv1d = d_inner * conv_kernel + d_inner
    # per-head biases / parameters
    dt_bias = num_heads
    a_log = num_heads
    d_param = num_heads
    # inner norm (RMSNorm over d_inner)
    inner_norm = d_inner
    # out_proj: d_inner → hidden
    out_proj = d_inner * hidden

    return in_proj + conv1d + dt_bias + a_log + d_param + inner_norm + out_proj


def _count_hybrid_params(
    config: dict, attn_type: AttentionType, mapping: MoEFieldMapping
) -> ParamCountResult:
    """Count params for hybrid architectures (Mamba2 + Attention + MoE)."""
    hidden = _require(config, "hidden_size")
    vocab = _require(config, "vocab_size")
    n_layers = _require(config, "num_hidden_layers")
    n_active_experts = _require(config, "num_experts_per_tok")
    n_proj = mapping.num_mlp_projections

    pattern = config.get("hybrid_override_pattern", "")
    if len(pattern) != n_layers:
        raise ValueError(
            f"hybrid_override_pattern length ({len(pattern)}) != num_hidden_layers ({n_layers})"
        )

    # Count layer types
    num_mamba = 0
    num_attn = 0
    num_moe = 0
    num_dense_mlp = 0

    attn_fn = _mla_attention_params if attn_type == AttentionType.MLA else _gqa_attention_params

    total_layer_params = 0
    active_layer_params = 0
    routed_expert_params = 0

    for char in pattern:
        # Each block: 1 RMSNorm + mixer
        norm_params = hidden

        if char == "M":
            mixer_params = _mamba2_layer_params(config)
            total_layer_params += norm_params + mixer_params
            active_layer_params += norm_params + mixer_params
            num_mamba += 1
        elif char == "*":
            mixer_params = attn_fn(config)
            total_layer_params += norm_params + mixer_params
            active_layer_params += norm_params + mixer_params
            num_attn += 1
        elif char == "-":
            intermediate = _require(config, "intermediate_size")
            mixer_params = _dense_mlp_params(hidden, intermediate, num_projections=n_proj)
            total_layer_params += norm_params + mixer_params
            active_layer_params += norm_params + mixer_params
            num_dense_mlp += 1
        else:
            # E or any other char → MoE layer
            total_moe = _moe_layer_params(config, mapping)
            active_moe = _moe_layer_params(config, mapping, num_active_experts=n_active_experts)
            total_layer_params += norm_params + total_moe
            active_layer_params += norm_params + active_moe

            # Track routed expert params
            n_experts = _require(config, mapping.expert_count_key)
            expert_intermediate = _require(config, mapping.expert_intermediate_key)
            routed_expert_params += n_experts * _dense_mlp_params(
                hidden, expert_intermediate, num_projections=n_proj
            )
            num_moe += 1

    # Embedding, lm_head, final norm
    embed = vocab * hidden
    lm_head = 0 if config.get("tie_word_embeddings", False) else vocab * hidden
    final_norm = hidden

    total = embed + lm_head + total_layer_params + final_norm
    active = embed + lm_head + active_layer_params + final_norm

    return ParamCountResult(
        total_params=total,
        active_params=active,
        routed_expert_params=routed_expert_params,
        model_type=config.get("model_type", ""),
        num_layers=n_layers,
        num_moe_layers=num_moe,
        num_dense_layers=num_dense_mlp,
        num_mamba_layers=num_mamba,
        num_attention_layers=num_attn,
    )


def _count_interval_hybrid_params(
    config: dict, attn_type: AttentionType, mapping: MoEFieldMapping
) -> ParamCountResult:
    """Count params for interval-based hybrid architectures (linear attn + full attn + MoE).

    Uses ``full_attention_interval`` to decide which layers get full GQA attention
    vs Mamba2-style linear attention.  Every ``full_attention_interval``-th layer
    (1-indexed from the end of each group) uses full attention; the rest use linear.
    All layers have MoE FFN.
    """
    hidden = _require(config, "hidden_size")
    vocab = _require(config, "vocab_size")
    n_layers = _require(config, "num_hidden_layers")
    n_active_experts = _require(config, "num_experts_per_tok")
    interval = _require(config, "full_attention_interval")

    n_proj = mapping.num_mlp_projections

    gqa_per_layer = _gqa_attention_params(config)
    linear_per_layer = _qwen3_next_linear_attn_params(config)

    total_moe_per_layer = _moe_layer_params(config, mapping)
    active_moe_per_layer = _moe_layer_params(config, mapping, num_active_experts=n_active_experts)

    n_experts = _require(config, mapping.expert_count_key)
    expert_intermediate = _require(config, mapping.expert_intermediate_key)
    routed_per_moe_layer = n_experts * _dense_mlp_params(
        hidden, expert_intermediate, num_projections=n_proj
    )

    num_full_attn = 0
    num_linear_attn = 0
    total_layer_params = 0
    active_layer_params = 0
    routed_expert_params = 0

    for i in range(n_layers):
        norms = 2 * hidden  # input_layernorm + post_attention_layernorm

        if (i + 1) % interval == 0:
            attn_params = gqa_per_layer
            num_full_attn += 1
        else:
            attn_params = linear_per_layer
            num_linear_attn += 1

        total_layer_params += norms + attn_params + total_moe_per_layer
        active_layer_params += norms + attn_params + active_moe_per_layer
        routed_expert_params += routed_per_moe_layer

    embed = vocab * hidden
    lm_head = 0 if config.get("tie_word_embeddings", False) else vocab * hidden
    final_norm = hidden

    total = embed + lm_head + total_layer_params + final_norm
    active = embed + lm_head + active_layer_params + final_norm

    return ParamCountResult(
        total_params=total,
        active_params=active,
        routed_expert_params=routed_expert_params,
        model_type=config.get("model_type", ""),
        num_layers=n_layers,
        num_moe_layers=n_layers,  # all layers have MoE FFN
        num_dense_layers=0,
        num_mamba_layers=num_linear_attn,
        num_attention_layers=num_full_attn,
    )


# ---------------------------------------------------------------------------
# Main entry point — parameter counting
# ---------------------------------------------------------------------------


def count_params_from_config(raw_config: dict) -> ParamCountResult:
    """Count total and active parameters from a HuggingFace config.json.

    Raises ``ValueError`` for unknown ``model_type`` or missing required
    fields.
    """
    config = resolve_text_config(raw_config)
    model_type = config.get("model_type", "")

    if model_type not in KNOWN_ARCHITECTURES:
        raise ValueError(f"Unknown model_type='{model_type}'")

    attn_type, mapping = KNOWN_ARCHITECTURES[model_type]

    # Hybrid architectures (Mamba2 + Attention + MoE) have a per-layer pattern
    if "hybrid_override_pattern" in config:
        return _count_hybrid_params(config, attn_type, mapping)

    # Interval-based hybrid (e.g. Qwen3-Next: linear attn + full GQA at intervals)
    if config.get("full_attention_interval", 0) > 1:
        return _count_interval_hybrid_params(config, attn_type, mapping)

    # Core dimensions
    hidden = _require(config, "hidden_size")
    vocab = _require(config, "vocab_size")
    n_layers = _require(config, "num_hidden_layers")
    intermediate = _require(config, "intermediate_size")
    n_active_experts = _require(config, "num_experts_per_tok")

    # Dense vs MoE layer split
    dense_layers = _require(config, mapping.dense_layers_key) if mapping.dense_layers_key else 0
    moe_layers = n_layers - dense_layers

    # MTP module count (optional)
    mtp_count = 0
    if mapping.mtp_key:
        mtp_count = config.get(mapping.mtp_key, 0) or 0

    # Attention params (same for every layer)
    attn_fn = _mla_attention_params if attn_type == AttentionType.MLA else _gqa_attention_params
    attn_per_layer = attn_fn(config)

    # Layer norms per layer (input_layernorm + post_attention_layernorm)
    norms_per_layer = 2 * hidden

    # --- Total params ---
    embed = vocab * hidden
    lm_head = 0 if config.get("tie_word_embeddings", False) else vocab * hidden
    final_norm = hidden

    # Routed expert params (the quantizable bulk in mixed-precision checkpoints)
    n_experts = _require(config, mapping.expert_count_key)
    expert_intermediate = _require(config, mapping.expert_intermediate_key)
    routed_expert_params = moe_layers * n_experts * _dense_mlp_params(hidden, expert_intermediate)

    # DSA indexer overhead (0 for non-DSA architectures)
    dsa_per_layer = _dsa_indexer_params(config)

    dense_block = attn_per_layer + _dense_mlp_params(hidden, intermediate) + norms_per_layer + dsa_per_layer
    moe_block = attn_per_layer + _moe_layer_params(config, mapping) + norms_per_layer + dsa_per_layer

    total = embed + lm_head + dense_layers * dense_block + moe_layers * moe_block + final_norm

    # MTP estimate: projection + norms per module
    mtp_per_module = 0
    if mtp_count > 0:
        mtp_per_module = 2 * hidden * hidden + 4 * hidden
        total += mtp_count * mtp_per_module

    # --- Active params (only num_experts_per_tok routed experts active) ---
    active_moe_block = (
        attn_per_layer
        + _moe_layer_params(config, mapping, num_active_experts=n_active_experts)
        + norms_per_layer
        + dsa_per_layer
    )
    active = (
        embed + lm_head + dense_layers * dense_block + moe_layers * active_moe_block + final_norm
    )
    if mtp_count > 0:
        active += mtp_count * mtp_per_module

    return ParamCountResult(
        total_params=total,
        active_params=active,
        routed_expert_params=routed_expert_params,
        model_type=model_type,
        num_layers=n_layers,
        num_moe_layers=moe_layers,
        num_dense_layers=dense_layers,
    )


# ---------------------------------------------------------------------------
# Precision detection
# ---------------------------------------------------------------------------


def detect_precision(raw_config: dict) -> PrecisionInfo:
    """Detect storage precision from config.json.

    Raises ``ValueError`` for unknown dtypes or quant methods.
    """
    config = resolve_text_config(raw_config)
    quant_config = config.get("quantization_config")

    if quant_config is None:
        # No quantization — use torch_dtype (or the "dtype" alias)
        dtype = config.get("torch_dtype") or config.get("dtype")
        dtype_map = {
            "bfloat16": (2.0, "BF16"),
            "float16": (2.0, "FP16"),
            "float32": (4.0, "FP32"),
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unknown torch_dtype='{dtype}'")
        bpp, label = dtype_map[dtype]
        return PrecisionInfo(bytes_per_param=bpp, dtype_str=label)

    method = quant_config.get("quant_method", "")

    if method == "fp8":
        return PrecisionInfo(bytes_per_param=1.0, dtype_str="FP8")

    if method == "compressed-tensors":
        groups = quant_config.get("config_groups", {})
        for group in groups.values():
            weights = group.get("weights", {})
            num_bits = weights.get("num_bits")
            w_type = weights.get("type")
            if num_bits and w_type:
                ignore_list = quant_config.get("ignore", [])
                is_mixed = bool(ignore_list)
                return PrecisionInfo(
                    bytes_per_param=num_bits / 8,
                    dtype_str=f"{w_type.upper()}{num_bits}",
                    is_mixed=is_mixed,
                )
        raise ValueError("Could not parse compressed-tensors config_groups")

    raise ValueError(f"Unknown quant_method='{method}'")
