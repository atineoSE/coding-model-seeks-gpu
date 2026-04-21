"""Tests for the LiteLLM API pricing source."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.sources.litellm_source import (
    LITELLM_ID_MAP,
    MODEL_LAB_MAP,
    PROVIDER_EXCLUDE_PREFIXES,
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
    "claude-opus-4-6-20250901": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "cache_creation_input_token_cost": 0.00001875,
        "cache_read_input_token_cost": 0.0000015,
        "context_window": 200000,
        "max_output_tokens": 32000,
    },
    "gpt-5.4": {
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00003,
        "cache_creation_input_token_cost": None,
        "cache_read_input_token_cost": None,
        "context_window": 128000,
        "max_output_tokens": 16384,
    },
    "gemini/gemini-3.1-pro": {
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000006,
        "cache_creation_input_token_cost": None,
        "cache_read_input_token_cost": None,
        "context_window": 1000000,
        "max_output_tokens": 8192,
    },
    # Cloud-routed variants that should be excluded
    "bedrock/claude-opus-4-6-20250901": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
    },
    "vertex_ai/gemini/gemini-3.1-pro": {
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000006,
    },
}


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
        # Qwen3.6-Plus is not in MODEL_LAB_MAP so it shouldn't appear
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
        assert entry["litellm_id"] == "claude-opus-4-6-20250901"
        assert entry["input_cost_per_token"] == 0.000015

    def test_excludes_cloud_routed_variants(self):
        best_models = {"anthropic": "claude-opus-4-6"}
        response = self._make_response(SAMPLE_LITELLM_JSON)

        with patch("pipeline.sources.litellm_source.urllib.request.urlopen", return_value=response):
            result = fetch_api_pricing(best_models)

        # Should use the direct key, not bedrock/
        assert "claude-opus-4-6" in result
        assert result["claude-opus-4-6"]["litellm_id"] == "claude-opus-4-6-20250901"

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


class TestModelLabMap:
    def test_covers_all_american_lab_closed_models(self):
        expected = {
            "claude-opus-4-6", "claude-opus-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6",
            "GPT-5.4", "GPT-5.2", "GPT-5.2-Codex",
            "Gemini-3.1-Pro", "Gemini-3-Pro", "Gemini-3-Flash",
        }
        assert expected.issubset(set(MODEL_LAB_MAP.keys()))

    def test_lab_values_are_valid(self):
        valid_labs = {"anthropic", "openai", "google"}
        assert all(lab in valid_labs for lab in MODEL_LAB_MAP.values())


class TestLitellmIdMap:
    def test_covers_current_best_models(self):
        expected_models = {"claude-opus-4-6", "GPT-5.4", "Gemini-3.1-Pro"}
        assert expected_models.issubset(set(LITELLM_ID_MAP.keys()))

    def test_no_cloud_routed_values(self):
        for key, value in LITELLM_ID_MAP.items():
            for prefix in PROVIDER_EXCLUDE_PREFIXES:
                assert not value.startswith(prefix), (
                    f"LITELLM_ID_MAP['{key}'] = '{value}' starts with excluded prefix '{prefix}'"
                )
