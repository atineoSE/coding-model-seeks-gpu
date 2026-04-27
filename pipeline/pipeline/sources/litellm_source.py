"""Fetch API pricing data from LiteLLM for the best closed model per American lab."""

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

# Maps OpenHands canonical model name → lab ("anthropic" | "openai" | "google").
# Covers all known closed-API models from American labs appearing in OpenHands Index.
MODEL_LAB_MAP: dict[str, str] = {
    # Anthropic
    "claude-opus-4-6": "anthropic",
    "claude-opus-4-5": "anthropic",
    "claude-sonnet-4-5": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    # OpenAI
    "GPT-5.4": "openai",
    "GPT-5.2": "openai",
    "GPT-5.2-Codex": "openai",
    # Google
    "Gemini-3.1-Pro": "google",
    "Gemini-3-Pro": "google",
    "Gemini-3-Flash": "google",
}

# Manually curated: OpenHands canonical name → LiteLLM dict key.
# Only covers the three current best-per-lab models; add new entries as models rotate.
# When a model's key cannot be found, a warning is logged and a notification is sent.
# See UPDATE-MODEL.md → "Updating API Pricing Mapping" for instructions.
LITELLM_ID_MAP: dict[str, str] = {
    "claude-opus-4-6": "claude-opus-4-6",
    "GPT-5.4": "gpt-5.4",
    "Gemini-3.1-Pro": "gemini-3.1-pro-preview",
}

PROVIDER_EXCLUDE_PREFIXES = ["bedrock/", "vertex_ai/", "azure/", "sagemaker/"]


def find_best_models_per_lab(benchmarks: list[dict]) -> dict[str, str]:
    """Return {lab: best_model_name} from the latest snapshot benchmarks.

    Considers only models with openness == "closed_api_available" that appear
    in MODEL_LAB_MAP. For each lab, picks the model with the highest "overall"
    benchmark score (falls back to any benchmark if "overall" is absent).
    """
    # lab → (has_overall, score, model_name)
    best: dict[str, tuple[bool, float, str]] = {}

    for entry in benchmarks:
        model_name = entry.get("model_name", "")
        if model_name not in MODEL_LAB_MAP:
            continue
        if entry.get("openness") != "closed_api_available":
            continue

        lab = MODEL_LAB_MAP[model_name]
        score = entry.get("score")
        if score is None:
            continue

        is_overall = entry.get("benchmark_name", "") == "overall"
        current = best.get(lab)

        if current is None:
            best[lab] = (is_overall, score, model_name)
        else:
            cur_is_overall, cur_score, _ = current
            # Prefer "overall" over non-overall; within same type, prefer higher score
            if (not cur_is_overall and is_overall) or (
                cur_is_overall == is_overall and score > cur_score
            ):
                best[lab] = (is_overall, score, model_name)

    return {lab: model_name for lab, (_, _, model_name) in best.items()}


def fetch_api_pricing(
    best_models: dict[str, str],
) -> dict[str, dict]:
    """Fetch LiteLLM pricing for the given best models.

    Args:
        best_models: {lab: openhands_model_name} as returned by find_best_models_per_lab.

    Returns:
        {openhands_model_name: {all LiteLLM fields..., "model_name", "lab", "litellm_id"}}

    Models whose LiteLLM key cannot be found are omitted; a warning is logged for each.
    """
    logger.info("Fetching LiteLLM pricing from %s", LITELLM_URL)
    with urllib.request.urlopen(LITELLM_URL, timeout=30) as resp:  # noqa: S310
        raw = json.loads(resp.read().decode())

    # Strip cloud-routed variants — keep direct-access keys only
    direct_pricing: dict[str, dict] = {
        key: value
        for key, value in raw.items()
        if not any(key.startswith(prefix) for prefix in PROVIDER_EXCLUDE_PREFIXES)
    }

    result: dict[str, dict] = {}
    for lab, model_name in best_models.items():
        litellm_key = LITELLM_ID_MAP.get(model_name, model_name.lower())
        entry = direct_pricing.get(litellm_key)
        if entry is None:
            logger.warning(
                "LiteLLM pricing key not found for '%s' (tried '%s'). "
                "Update LITELLM_ID_MAP in litellm_source.py.",
                model_name,
                litellm_key,
            )
            continue

        result[model_name] = {
            **entry,
            "model_name": model_name,
            "lab": lab,
            "litellm_id": litellm_key,
            # Normalise: LiteLLM uses max_input_tokens; frontend expects context_window
            "context_window": entry.get("context_window") or entry.get("max_input_tokens"),
        }

    logger.info("Fetched pricing for %d/%d models", len(result), len(best_models))
    return result
