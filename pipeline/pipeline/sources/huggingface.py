"""HuggingFace model source — deterministic via HF API + config.json."""

import logging
import time

import httpx

from pipeline.enrichment import ModelSpec
from pipeline.sources.param_counter import (
    KNOWN_ARCHITECTURES,
    AttentionType,
    count_params_from_config,
    detect_precision,
    resolve_text_config,
)

logger = logging.getLogger(__name__)

# Manual mapping of benchmark model names to HuggingFace repo IDs.
# This bridges the gap between OpenHands leaderboard names and HF repos.
# Only open-source / open-weight models are included; closed-source models
# (GPT, Claude, Gemini) are skipped since they can't be self-hosted.
MODEL_NAME_TO_HF_ID: dict[str, str] = {
    "DeepSeek-V3.2-Reasoner": "deepseek-ai/DeepSeek-V3.2-Speciale",
    "GLM-4.7": "zai-org/GLM-4.7",
    "Kimi-K2.5": "moonshotai/Kimi-K2.5",
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Thinking",
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
    "MiniMax-M2.1": "MiniMaxAI/MiniMax-M2.1",
    "Qwen3-Coder-480B": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Nemotron-3-Nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
}


def fetch_hf_config(hf_id: str) -> dict | None:
    """Fetch a HuggingFace model config.json for architecture details."""
    url = f"https://huggingface.co/{hf_id}/raw/main/config.json"
    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        config = response.json()
        logger.info("Fetched config.json for %s", hf_id)
        return config
    except Exception:
        logger.debug("No config.json available for %s", hf_id)
        return None


def fetch_hf_param_count(hf_id: str) -> float | None:
    """Fetch total parameter count from the HF API (safetensors metadata).

    This is the authoritative param count shown on the HF model page.
    Returns total params in billions, or None if unavailable.
    """
    url = f"https://huggingface.co/api/models/{hf_id}"
    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        total = data.get("safetensors", {}).get("parameters", {})
        if not total:
            return None
        # Sum all dtypes to get total param count
        total_params = sum(total.values())
        params_b = round(total_params / 1e9, 1)
        logger.info("HF API param count for %s: %.1fB", hf_id, params_b)
        return params_b
    except Exception:
        logger.debug("Could not fetch HF API param count for %s", hf_id)
        return None


def fetch_model(model_name: str, hf_id: str) -> ModelSpec:
    """Build a ModelSpec from the HF API and config.json.

    Uses config-based parameter counting as the primary source for the true
    logical parameter count (safetensors element counts can be wrong for
    packed formats like INT4).  Raises if both paths fail.

    Args:
        model_name: Display name of the model.
        hf_id: HuggingFace repo ID (e.g., "deepseek-ai/DeepSeek-V3.2").

    Raises:
        RuntimeError: If config.json is unavailable.
        ValueError: If config-based param counting fails and safetensors
            data is also missing.
    """
    # 1. Fetch config.json first (required for all paths)
    config = fetch_hf_config(hf_id)
    if not config:
        raise RuntimeError(f"No config.json available for {hf_id}")

    # 2. Fetch safetensors element count (shown on HF model page)
    published_param_count_b = fetch_hf_param_count(hf_id)

    # 3. Run config-based counting (true logical param count)
    config_result = None
    try:
        config_result = count_params_from_config(config)
    except ValueError as e:
        logger.warning("Config-based param count failed for %s: %s", hf_id, e)
        if published_param_count_b is None:
            raise  # Pipeline fails if both paths unavailable

    # 4. Learnable params from config; fall back to safetensors
    if config_result is not None:
        learnable_params_b = round(config_result.total_params / 1e9, 1)
    elif published_param_count_b is not None:
        learnable_params_b = published_param_count_b
    else:
        raise ValueError(f"No param count available for {hf_id}")

    # 5. Active params from config result
    active_params_b = None
    if config_result:
        active_params_b = round(config_result.active_params / 1e9, 1)

    # 5b. Routed expert params (the INT4-quantized bulk for mixed-precision models)
    routed_expert_params_b = None
    if config_result and config_result.routed_expert_params > 0:
        routed_expert_params_b = round(config_result.routed_expert_params / 1e9, 1)

    # 6. MoE detection from config result
    is_moe = config_result.num_moe_layers > 0 if config_result else False

    # 7. Resolve max_position_embeddings from effective config
    #    (fixes Kimi-K2.5 where text_config wrapper hides the field)
    effective_config = resolve_text_config(config)
    context_length = effective_config.get("max_position_embeddings")

    # 8. Extract KV cache architecture fields (required for all known architectures)
    model_type = effective_config.get("model_type", "")
    if model_type not in KNOWN_ARCHITECTURES:
        raise ValueError(
            f"Unknown model_type='{model_type}' for {hf_id} — "
            f"cannot determine KV cache architecture"
        )

    attn_type, _ = KNOWN_ARCHITECTURES[model_type]
    attention_type_str = attn_type.value.upper()  # "MLA" or "GQA"

    num_hidden_layers = effective_config.get("num_hidden_layers")
    if num_hidden_layers is None:
        raise ValueError(f"Missing 'num_hidden_layers' in config for {hf_id}")
    num_hidden_layers = int(num_hidden_layers)

    num_kv_heads: int | None = None
    head_dim_val: int | None = None
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None

    if attn_type == AttentionType.GQA:
        num_kv_heads = effective_config.get("num_key_value_heads")
        if num_kv_heads is None:
            raise ValueError(f"Missing 'num_key_value_heads' in GQA config for {hf_id}")
        num_kv_heads = int(num_kv_heads)

        explicit_head_dim = effective_config.get("head_dim")
        if explicit_head_dim is not None:
            head_dim_val = int(explicit_head_dim)
        else:
            hidden = effective_config.get("hidden_size")
            n_heads = effective_config.get("num_attention_heads")
            if hidden is None or n_heads is None or int(n_heads) == 0:
                raise ValueError(
                    f"Cannot compute head_dim for {hf_id}: "
                    f"need 'head_dim' or 'hidden_size'/'num_attention_heads'"
                )
            head_dim_val = int(hidden) // int(n_heads)

    elif attn_type == AttentionType.MLA:
        kv_lora_rank = effective_config.get("kv_lora_rank")
        if kv_lora_rank is None:
            raise ValueError(f"Missing 'kv_lora_rank' in MLA config for {hf_id}")
        kv_lora_rank = int(kv_lora_rank)

        qk_rope_head_dim = effective_config.get("qk_rope_head_dim")
        if qk_rope_head_dim is None:
            raise ValueError(f"Missing 'qk_rope_head_dim' in MLA config for {hf_id}")
        qk_rope_head_dim = int(qk_rope_head_dim)

    # 9. Detect precision
    precision_str = None
    try:
        precision_info = detect_precision(config)
        if precision_info.is_mixed:
            precision_str = precision_info.dtype_str
        else:
            precision_str = precision_info.dtype_str
    except ValueError as e:
        logger.warning("Precision detection failed for %s: %s", hf_id, e)

    return ModelSpec(
        model_name=model_name,
        published_param_count_b=published_param_count_b,
        learnable_params_b=learnable_params_b,
        active_params_b=active_params_b,
        architecture="MoE" if is_moe else "Dense",
        context_length=context_length,
        precision=precision_str,
        routed_expert_params_b=routed_expert_params_b,
        attention_type=attention_type_str,
        num_hidden_layers=num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim_val,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        hf_model_id=hf_id,
    )


def fetch_all_models(
    model_map: dict[str, str] | None = None,
    delay: float = 2.0,
) -> list[ModelSpec]:
    """Fetch all models from the mapping.

    Args:
        model_map: Dict of model_name -> hf_id. Uses default mapping if None.
        delay: Seconds to wait between requests to avoid rate limiting.

    Returns:
        List of successfully enriched ModelSpec objects.
    """
    if model_map is None:
        model_map = MODEL_NAME_TO_HF_ID

    results = []
    failed = []
    total = len(model_map)

    for i, (name, hf_id) in enumerate(model_map.items(), 1):
        logger.info("Fetching model %d/%d: %s (%s)", i, total, name, hf_id)

        try:
            spec = fetch_model(name, hf_id)
            results.append(spec)
            logger.info(
                "  -> %s: learnable=%.1fB published=%.1fB (active=%.1fB), %s, %s",
                spec.model_name,
                spec.learnable_params_b or 0,
                spec.published_param_count_b or 0,
                spec.active_params_b or 0,
                spec.architecture,
                spec.precision or "unknown",
            )
        except Exception as e:
            failed.append(name)
            logger.warning("  -> Failed to enrich %s: %s", name, e)

        if i < total:
            time.sleep(delay)

    logger.info("Successfully fetched %d/%d models", len(results), total)

    if failed:
        raise RuntimeError(f"Failed to enrich {len(failed)}/{total} models: {', '.join(failed)}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test with a single benchmark model
    spec = fetch_model("GLM-4.7", "zai-org/GLM-4.7")
    print(f"\n{spec.model_name}:")
    print(f"  Published Params: {spec.published_param_count_b}B")
    print(f"  Learnable Params: {spec.learnable_params_b}B")
    print(f"  Active: {spec.active_params_b}B")
    print(f"  Architecture: {spec.architecture}")
    print(f"  Precision: {spec.precision}")
    print(f"  Context: {spec.context_length}")
