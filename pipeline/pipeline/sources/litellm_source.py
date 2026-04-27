"""Fetch API pricing data from LiteLLM for the best closed model per American lab."""

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

# Maps model name prefix (lowercase) → lab.
# Models are matched case-insensitively, so new models are picked up automatically.
LAB_PATTERNS: list[tuple[str, str]] = [
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("gemini-", "google"),
]

# Fallback: OpenHands canonical name → LiteLLM dict key.
# Only needed for models where model_name.lower() doesn't match the LiteLLM key directly
# (e.g. "Gemini-3.1-Pro" → "gemini-3.1-pro-preview" due to a -preview suffix).
# When a model's key cannot be found via either method, a warning is logged.
LITELLM_ID_MAP: dict[str, str] = {
    "Gemini-3.1-Pro": "gemini-3.1-pro-preview",
}

PROVIDER_EXCLUDE_PREFIXES = ["bedrock/", "vertex_ai/", "azure/", "sagemaker/"]


def _get_lab_for_model(model_name: str) -> str | None:
    """Return the lab for a model name based on name prefix, or None if unrecognised."""
    lower = model_name.lower()
    for prefix, lab in LAB_PATTERNS:
        if lower.startswith(prefix):
            return lab
    return None


def find_best_models_per_lab(benchmarks: list[dict]) -> dict[str, str]:
    """Return {lab: best_model_name} from the latest snapshot benchmarks.

    Considers only closed-API models whose names match a known lab prefix.
    For each lab, picks the model with the highest "overall" benchmark score
    (falls back to any benchmark if "overall" is absent).
    """
    # lab → (has_overall, score, model_name)
    best: dict[str, tuple[bool, float, str]] = {}

    for entry in benchmarks:
        model_name = entry.get("model_name", "")
        lab = _get_lab_for_model(model_name)
        if lab is None:
            continue
        if entry.get("openness") != "closed_api_available":
            continue

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

    Key resolution order:
      1. model_name.lower() — matches most models directly (e.g. "GPT-5.4" → "gpt-5.4")
      2. model_name.lower() + "-preview" — handles preview variants (e.g. "Gemini-3.1-Pro" → "gemini-3.1-pro-preview")
      3. LITELLM_ID_MAP[model_name] — last resort for fully custom key mappings

    Models whose LiteLLM key cannot be found are omitted; a warning is logged for each.

    Args:
        best_models: {lab: openhands_model_name} as returned by find_best_models_per_lab.

    Returns:
        {openhands_model_name: {all LiteLLM fields..., "model_name", "lab", "litellm_id"}}
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
        # Step 1: try lowercase (e.g. "GPT-5.4" → "gpt-5.4")
        litellm_key = model_name.lower()
        if litellm_key not in direct_pricing:
            # Step 2: try lowercase + "-preview" (e.g. "gemini-3.1-pro" → "gemini-3.1-pro-preview")
            preview_key = litellm_key + "-preview"
            if preview_key in direct_pricing:
                litellm_key = preview_key
            else:
                # Step 3: fall back to LITELLM_ID_MAP for fully custom mappings
                litellm_key = LITELLM_ID_MAP.get(model_name, litellm_key)

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
