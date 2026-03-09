"""Tests for license metadata in the pipeline.

Covers the MODEL_LICENSE_INFO mapping, fetch_model integration (mocked HTTP),
and the pipeline-to-frontend data contract (ModelSpec fields match
the TypeScript Model interface).
"""

from unittest.mock import patch

import pytest

from pipeline.enrichment import ModelSpec
from pipeline.sources.huggingface import (
    MODEL_LICENSE_INFO,
    MODEL_NAME_TO_HF_ID,
    fetch_model,
)

# Re-use existing test configs for integration tests
from tests.test_param_counter import (
    DEEPSEEK_V32_CONFIG,
    GLM_47_CONFIG,
    QWEN3_CODER_CONFIG,
)


# ===================================================================
# MODEL_LICENSE_INFO mapping — unit tests
# ===================================================================


class TestModelLicenseInfo:
    """Tests for the explicit license mapping."""

    def test_all_models_have_license_info(self):
        """Every model in MODEL_NAME_TO_HF_ID must have a license entry."""
        for model_name, hf_id in MODEL_NAME_TO_HF_ID.items():
            assert hf_id in MODEL_LICENSE_INFO, (
                f"Missing license info for {model_name} ({hf_id}). "
                f"Add an entry to MODEL_LICENSE_INFO."
            )

    def test_license_entries_have_name_and_url(self):
        """Each license entry must have a non-empty name and URL."""
        for hf_id, (name, url) in MODEL_LICENSE_INFO.items():
            assert name, f"Empty license_name for {hf_id}"
            assert url, f"Empty license_url for {hf_id}"

    def test_minimax_m25_license(self):
        name, url = MODEL_LICENSE_INFO["MiniMaxAI/MiniMax-M2.5"]
        assert name == "MiniMax Model License"
        assert "LICENSE-MODEL" in url

    def test_deepseek_mit_license(self):
        name, url = MODEL_LICENSE_INFO["deepseek-ai/DeepSeek-V3.2-Speciale"]
        assert name == "MIT"
        assert "deepseek-ai/DeepSeek-V3.2-Speciale" in url

    def test_qwen_apache_license(self):
        name, url = MODEL_LICENSE_INFO["Qwen/Qwen3-Coder-480B-A35B-Instruct"]
        assert name == "Apache 2.0"


# ===================================================================
# fetch_model integration — license fields populated from mapping
# ===================================================================


def _mock_fetch(model_name, hf_id, config):
    """Call fetch_model with mocked config.json."""
    with patch("pipeline.sources.huggingface.fetch_hf_config", return_value=config):
        return fetch_model(model_name, hf_id)


class TestFetchModelLicenseIntegration:
    """Integration tests: fetch_model populates license fields from mapping."""

    def test_mit_license_from_mapping(self):
        spec = _mock_fetch(
            "DS", "deepseek-ai/DeepSeek-V3.2-Speciale", DEEPSEEK_V32_CONFIG
        )
        assert spec.license_name == "MIT"
        assert "deepseek-ai/DeepSeek-V3.2-Speciale" in spec.license_url

    def test_apache_license_from_mapping(self):
        spec = _mock_fetch(
            "Q3", "Qwen/Qwen3-Coder-480B-A35B-Instruct", QWEN3_CODER_CONFIG
        )
        assert spec.license_name == "Apache 2.0"
        assert "Qwen3-Coder-480B-A35B-Instruct" in spec.license_url

    def test_missing_license_raises(self):
        """fetch_model should fail if hf_id is not in MODEL_LICENSE_INFO."""
        with pytest.raises(ValueError, match="Missing license info"):
            _mock_fetch("Unknown", "org/unknown-model", GLM_47_CONFIG)

    def test_license_fields_coexist_with_existing_fields(self):
        """License fields should not break existing ModelSpec fields."""
        spec = _mock_fetch(
            "DS", "deepseek-ai/DeepSeek-V3.2-Speciale", DEEPSEEK_V32_CONFIG
        )
        # Existing fields still work
        assert spec.model_name == "DS"
        assert spec.attention_type == "MLA"
        assert spec.kv_lora_rank == 512
        # License fields present
        assert spec.license_name == "MIT"


# ===================================================================
# Data contract: ModelSpec <-> TypeScript Model interface
# ===================================================================


# The TypeScript Model interface fields (from web/src/types/index.ts).
# This acts as a contract test — if either side changes, this test breaks.
TYPESCRIPT_MODEL_FIELDS = {
    "model_name",
    "learnable_params_b",
    "active_params_b",
    "architecture",
    "context_length",
    "precision",
    "routed_expert_params_b",
    "attention_type",
    "num_hidden_layers",
    "num_kv_layers",
    "num_kv_heads",
    "head_dim",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "hf_model_id",
    "license_name",
    "license_url",
}


class TestDataContract:
    """Verify that ModelSpec fields match what the frontend expects."""

    def test_modelspec_fields_match_typescript(self):
        """ModelSpec should have exactly the fields the TS Model interface expects."""
        spec_fields = set(ModelSpec.model_fields.keys())
        assert spec_fields == TYPESCRIPT_MODEL_FIELDS, (
            f"ModelSpec fields differ from TypeScript Model interface.\n"
            f"  In Python but not TS: {spec_fields - TYPESCRIPT_MODEL_FIELDS}\n"
            f"  In TS but not Python: {TYPESCRIPT_MODEL_FIELDS - spec_fields}"
        )

    def test_modelspec_json_roundtrip_includes_license(self):
        """model_dump() output should include license fields (used by JSON exporter)."""
        spec = ModelSpec(
            model_name="TestModel",
            license_name="MIT",
            license_url="https://example.com/LICENSE",
        )
        dumped = spec.model_dump()
        assert dumped["license_name"] == "MIT"
        assert dumped["license_url"] == "https://example.com/LICENSE"

    def test_modelspec_json_roundtrip_null_license(self):
        """License fields should be null when not set."""
        spec = ModelSpec(model_name="TestModel")
        dumped = spec.model_dump()
        assert dumped["license_name"] is None
        assert dumped["license_url"] is None
