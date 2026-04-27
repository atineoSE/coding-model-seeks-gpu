"""Tests for the LiteLLM API pricing source."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.sources.litellm_source import (
    LAB_PATTERNS,
    LITELLM_ID_MAP,
    PROVIDER_EXCLUDE_PREFIXES,
    _get_lab_for_model,
    fetch_api_pricing,
    find_best_models_per_lab,
)


SAMPLE_BENCHMARKS = [
    {
        "model_name": "claude-opus-4-6",
        "benchmark_name": "overall",
        "score": 66.7,
        "openness": "closed_api_available",
    },
    {
        "model_name": "claude-opus-4-5",
        "benchmark_name": "overall",
        "score": 60.6,
        "openness": "closed_api_available",
    },
    {
        "model_name": "GPT-5.4",
        "benchmark_name": "overall",
        "score": 63.8,
        "openness": "closed_api_available",
    },
    {
        "model_name": "GPT-5.2",
        "benchmark_name": "overall",
        "score": 56.3,
        "openness": "closed_api_available",
    },
    {
        "model_name": "Gemini-3.1-Pro",
        "benchmark_name": "overall",
        "score": 55.7,
        "openness": "closed_api_available",
    },
    {
        "model_name": "GLM-5",
        "benchmark_name": "overall",
        "score": 49.4,
        "openness": "open_weights",
    },
    {
        "model_name": "Qwen3.6-Plus",
        "benchmark_name": "overall",
        "score": 57.9,
        "openness": "closed_api_available",
    },
]

SAMPLE_LITELLM_JSON = {
    "claude-opus-4-6": {
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000025,
        "cache_creation_input_token_cost": 0.000006,
        "cache_read_input_token_cost": 0.0000005,
        "context_window": 1000000,
        "max_output_tokens": 32000,
    },
    "gpt-5.4": {
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.000015,
        "cache_creation_input_token_cost": None,
        "cache_read_input_token_cost": 0.00000025,
        "max_input_tokens": 1050000,  # LiteLLM uses max_input_tokens, not context_window
        "max_output_tokens": 128000,
    },
    "gemini-3.1-pro-preview": {
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
        "cache_creation_input_token_cost": None,
        "cache_read_input_token_cost": 0.0000002,
        "max_input_tokens": 1048576,
        "max_output_tokens": 8192,
    },
    # Cloud-routed variants that should be excluded
    "bedrock/claude-opus-4-6": {
        "input_cost_per_token": 0.000005,
        "output_cost_per_token": 0.000025,
    },
    "vertex_ai/gemini-3.1-pro-preview": {
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000008,
    },
}


class TestGetLabForModel:
    def test_claude_prefix_returns_anthropic(self):
        assert _get_lab_for_model("claude-opus-4-6") == "anthropic"
        assert _get_lab_for_model("claude-sonnet-4-5") == "anthropic"

    def test_gpt_prefix_returns_openai(self):
        assert _get_lab_for_model("GPT-5.4") == "openai"
        assert _get_lab_for_model("gpt-5.2-codex") == "openai"

    def test_gemini_prefix_returns_google(self):
        assert _get_lab_for_model("Gemini-3.1-Pro") == "google"
        assert _get_lab_for_model("gemini-3-flash") == "google"

    def test_prefix_matching_is_case_insensitive(self):
        assert _get_lab_for_model("CLAUDE-OPUS-4-6") == "anthropic"
        assert _get_lab_for_model("GPT-5.4") == "openai"
        assert _get_lab_for_model("GEMINI-3-PRO") == "google"

    def test_unknown_prefix_returns_none(self):
        assert _get_lab_for_model("Qwen3.6-Plus") is None
        assert _get_lab_for_model("GLM-5") is None
        assert _get_lab_for_model("") is None

    def test_lab_patterns_covers_three_labs(self):
        labs = {lab for _, lab in LAB_PATTERNS}
        assert labs == {"anthropic", "openai", "google"}


class TestFindBestModelsPerLab:
    def test_picks_highest_score_per_lab(self):
        result = find_best_models_per_lab(SAMPLE_BENCHMARKS)
        assert result["anthropic"] == "claude-opus-4-6"
        assert result["openai"] == "GPT-5.4"
        assert result["google"] == "Gemini-3.1-Pro"

    def test_ignores_open_weights(self):
        result = find_best_models_per_lab(SAMPLE_BENCHMARKS)
        assert "GLM-5" not in result.values()

    def test_ignores_non_american_labs(self):
        result = find_best_models_per_lab(SAMPLE_BENCHMARKS)
        # Qwen3.6-Plus doesn't match any lab prefix so it shouldn't appear
        model_names = set(result.values())
        assert "Qwen3.6-Plus" not in model_names

    def test_empty_benchmarks(self):
        assert find_best_models_per_lab([]) == {}

    def test_no_closed_models(self):
        benchmarks = [
            {"model_name": "GLM-5", "benchmark_name": "overall", "score": 50.0, "openness": "open_weights"}
        ]
        assert find_best_models_per_lab(benchmarks) == {}

    def test_missing_score_skipped(self):
        benchmarks = [
            {"model_name": "claude-opus-4-6", "benchmark_name": "overall", "openness": "closed_api_available"},
        ]
        result = find_best_models_per_lab(benchmarks)
        assert "anthropic" not in result

    def test_non_overall_benchmark_used_when_no_overall(self):
        benchmarks = [
            {
                "model_name": "claude-opus-4-6",
                "benchmark_name": "swe-bench",
                "score": 55.0,
                "openness": "closed_api_available",
            }
        ]
        result = find_best_models_per_lab(benchmarks)
        assert result.get("anthropic") == "claude-opus-4-6"

    def test_new_model_picked_up_without_map_update(self):
        """A future model matching a known prefix is automatically assigned to the right lab."""
        benchmarks = [
            {
                "model_name": "claude-opus-4-99",
                "benchmark_name": "overall",
                "score": 99.0,
                "openness": "closed_api_available",
            }
        ]
        result = find_best_models_per_lab(benchmarks)
        assert result.get("anthropic") == "claude-opus-4-99"


class TestFetchApiPricing:
    def _make_response(self, data):
        response = MagicMock()
        response.read.return_value = json.dumps(data).encode()
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        return response

    def test_returns_pricing_with_added_fields(self):
        best_models = {"anthropic": "claude-opus-4-6", "openai": "GPT-5.4", "google": "Gemini-3.1-Pro"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        assert "claude-opus-4-6" in result
        entry = result["claude-opus-4-6"]
        assert entry["model_name"] == "claude-opus-4-6"
        assert entry["lab"] == "anthropic"
        assert entry["litellm_id"] == "claude-opus-4-6"
        assert entry["input_cost_per_token"] == 0.000005

    def test_excludes_cloud_routed_variants(self):
        best_models = {"anthropic": "claude-opus-4-6"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        # Should use the direct key, not bedrock/
        assert "claude-opus-4-6" in result
        assert result["claude-opus-4-6"]["litellm_id"] == "claude-opus-4-6"

    def test_lowercase_key_resolves_directly(self):
        """GPT-5.4 → gpt-5.4 via lowercase, no LITELLM_ID_MAP needed."""
        best_models = {"openai": "GPT-5.4"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        assert "GPT-5.4" in result
        assert result["GPT-5.4"]["litellm_id"] == "gpt-5.4"

    def test_preview_suffix_tried_before_litellm_id_map(self):
        """Gemini-3.1-Pro → gemini-3.1-pro misses, then gemini-3.1-pro-preview hits."""
        best_models = {"google": "Gemini-3.1-Pro"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        assert "Gemini-3.1-Pro" in result
        assert result["Gemini-3.1-Pro"]["litellm_id"] == "gemini-3.1-pro-preview"

    def test_fallback_to_litellm_id_map_when_preview_also_misses(self):
        """When both lowercase and lowercase-preview miss, LITELLM_ID_MAP is used."""
        litellm_json = {
            **SAMPLE_LITELLM_JSON,
            "my-model-special": {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
        }
        # Remove gemini-3.1-pro-preview so -preview step fails
        litellm_json = {k: v for k, v in litellm_json.items() if k != "gemini-3.1-pro-preview"}

        from unittest.mock import patch as _patch
        import pipeline.sources.litellm_source as src
        original_map = src.LITELLM_ID_MAP.copy()
        src.LITELLM_ID_MAP["Gemini-3.1-Pro"] = "my-model-special"
        try:
            best_models = {"google": "Gemini-3.1-Pro"}
            response = self._make_response(litellm_json)
            with _patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
                result = fetch_api_pricing(best_models)
        finally:
            src.LITELLM_ID_MAP.clear()
            src.LITELLM_ID_MAP.update(original_map)

        assert "Gemini-3.1-Pro" in result
        assert result["Gemini-3.1-Pro"]["litellm_id"] == "my-model-special"

    def test_warns_on_missing_key(self, caplog):
        best_models = {"anthropic": "claude-opus-4-99"}  # not in LITELLM_ID_MAP or raw data
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        assert "claude-opus-4-99" not in result
        assert any("not found" in r.message.lower() for r in caplog.records)

    def test_all_three_labs_fetched(self):
        best_models = {"anthropic": "claude-opus-4-6", "openai": "GPT-5.4", "google": "Gemini-3.1-Pro"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        assert len(result) == 3
        labs = {v["lab"] for v in result.values()}
        assert labs == {"anthropic", "openai", "google"}

    def test_context_window_normalised_from_max_input_tokens(self):
        """LiteLLM uses max_input_tokens; pipeline must expose it as context_window."""
        best_models = {"openai": "GPT-5.4"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        assert result["GPT-5.4"]["context_window"] == 1050000

    def test_context_window_preserved_when_already_present(self):
        """Entries that already have context_window are not overwritten."""
        best_models = {"anthropic": "claude-opus-4-6"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        assert result["claude-opus-4-6"]["context_window"] == 1000000

    def test_partial_caching_openai_style(self):
        """cache_creation_input_token_cost=None with cache_read set should be preserved."""
        best_models = {"openai": "GPT-5.4"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        entry = result["GPT-5.4"]
        assert entry["cache_read_input_token_cost"] == 0.00000025
        assert entry["cache_creation_input_token_cost"] is None


class TestLitellmIdMap:
    def test_only_contains_models_needing_non_trivial_key_mapping(self):
        """LITELLM_ID_MAP should only hold models where lowercase doesn't match directly."""
        for openhands_name, litellm_key in LITELLM_ID_MAP.items():
            assert openhands_name.lower() != litellm_key, (
                f"'{openhands_name}' lowercases to its LiteLLM key — remove from LITELLM_ID_MAP"
            )

    def test_no_cloud_routed_values(self):
        for key, value in LITELLM_ID_MAP.items():
            for prefix in PROVIDER_EXCLUDE_PREFIXES:
                assert not value.startswith(prefix), (
                    f"LITELLM_ID_MAP['{key}'] = '{value}' starts with excluded prefix '{prefix}'"
                )
