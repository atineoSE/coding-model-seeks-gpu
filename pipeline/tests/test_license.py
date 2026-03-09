"""Tests for license metadata extraction from HuggingFace API.

Covers resolve_license_info logic, fetch_model integration (mocked HTTP),
and the pipeline-to-frontend data contract (ModelSpec fields match
the TypeScript Model interface).
"""

from unittest.mock import patch

import pytest

from pipeline.enrichment import ModelSpec
from pipeline.sources.huggingface import (
    CHOOSEALICENSE_BASE_URL,
    CUSTOM_LICENSE_DISPLAY_NAMES,
    SPDX_DISPLAY_NAMES,
    resolve_license_info,
    fetch_model,
)

# Re-use existing test configs for integration tests
from tests.test_param_counter import (
    DEEPSEEK_V32_CONFIG,
    GLM_47_CONFIG,
    QWEN3_CODER_CONFIG,
)


# ===================================================================
# resolve_license_info — unit tests
# ===================================================================


class TestResolveLicenseInfo:
    """Unit tests for license info resolution from HF API metadata."""

    def test_standard_mit_with_license_file(self):
        metadata = {
            "cardData": {"license": "mit"},
            "siblings": [{"rfilename": "LICENSE"}, {"rfilename": "config.json"}],
        }
        spdx, name, url = resolve_license_info(metadata, "deepseek-ai/DS")
        assert spdx == "mit"
        assert name == "MIT"
        assert url == "https://huggingface.co/deepseek-ai/DS/blob/main/LICENSE"

    def test_standard_mit_without_license_file(self):
        """Falls back to choosealicense dataset."""
        metadata = {
            "cardData": {"license": "mit"},
            "siblings": [{"rfilename": "config.json"}],
        }
        spdx, name, url = resolve_license_info(metadata, "zai-org/GLM")
        assert spdx == "mit"
        assert name == "MIT"
        assert url == f"{CHOOSEALICENSE_BASE_URL}/mit.md"

    def test_apache_with_license_link(self):
        metadata = {
            "cardData": {
                "license": "apache-2.0",
                "license_link": "https://huggingface.co/Qwen/model/blob/main/LICENSE",
            },
            "siblings": [{"rfilename": "LICENSE"}],
        }
        spdx, name, url = resolve_license_info(metadata, "Qwen/model")
        assert spdx == "apache-2.0"
        assert name == "Apache 2.0"
        # license_link takes priority over siblings
        assert url == "https://huggingface.co/Qwen/model/blob/main/LICENSE"

    def test_custom_modified_mit(self):
        metadata = {
            "cardData": {"license": "other", "license_name": "modified-mit"},
            "siblings": [{"rfilename": "LICENSE"}],
        }
        spdx, name, url = resolve_license_info(metadata, "moonshotai/Kimi")
        assert spdx == "other"
        assert name == "Modified MIT"
        assert url == "https://huggingface.co/moonshotai/Kimi/blob/main/LICENSE"

    def test_custom_nvidia_with_license_link(self):
        metadata = {
            "cardData": {
                "license": "other",
                "license_name": "nvidia-open-model-license",
                "license_link": "https://www.nvidia.com/license",
            },
            "siblings": [],
        }
        spdx, name, url = resolve_license_info(metadata, "nvidia/model")
        assert spdx == "other"
        assert name == "NVIDIA Open Model License"
        assert url == "https://www.nvidia.com/license"

    def test_custom_unknown_license_name_title_cased(self):
        """Unknown custom license names should be title-cased."""
        metadata = {
            "cardData": {"license": "other", "license_name": "some-custom-license"},
            "siblings": [],
        }
        spdx, name, url = resolve_license_info(metadata, "org/model")
        assert name == "Some Custom License"
        assert url is None  # no file, no link, not standard SPDX

    def test_missing_card_data(self):
        metadata = {"siblings": []}
        spdx, name, url = resolve_license_info(metadata, "org/model")
        assert spdx is None
        assert name is None
        assert url is None

    def test_empty_card_data(self):
        metadata = {"cardData": {}, "siblings": []}
        spdx, name, url = resolve_license_info(metadata, "org/model")
        assert spdx is None
        assert name is None
        assert url is None

    def test_url_priority_license_link_over_siblings(self):
        """license_link should take priority even when LICENSE file exists."""
        metadata = {
            "cardData": {
                "license": "other",
                "license_name": "modified-mit",
                "license_link": "https://github.com/org/repo/blob/main/LICENSE",
            },
            "siblings": [{"rfilename": "LICENSE"}],
        }
        _, _, url = resolve_license_info(metadata, "org/model")
        assert url == "https://github.com/org/repo/blob/main/LICENSE"

    def test_url_priority_siblings_over_choosealicense(self):
        """LICENSE file in repo should take priority over choosealicense fallback."""
        metadata = {
            "cardData": {"license": "mit"},
            "siblings": [{"rfilename": "LICENSE"}],
        }
        _, _, url = resolve_license_info(metadata, "org/model")
        assert url == "https://huggingface.co/org/model/blob/main/LICENSE"

    def test_all_spdx_display_names_mapped(self):
        """All entries in SPDX_DISPLAY_NAMES should produce the expected display name."""
        for spdx_id, expected_name in SPDX_DISPLAY_NAMES.items():
            metadata = {"cardData": {"license": spdx_id}, "siblings": []}
            _, name, _ = resolve_license_info(metadata, "org/model")
            assert name == expected_name, f"SPDX '{spdx_id}' should display as '{expected_name}'"

    def test_unknown_spdx_falls_back_to_code(self):
        """Unknown SPDX identifiers should use the raw code as display name."""
        metadata = {"cardData": {"license": "bsd-3-clause"}, "siblings": []}
        spdx, name, url = resolve_license_info(metadata, "org/model")
        assert spdx == "bsd-3-clause"
        assert name == "bsd-3-clause"
        assert url == f"{CHOOSEALICENSE_BASE_URL}/bsd-3-clause.md"

    def test_all_custom_display_names_mapped(self):
        """All entries in CUSTOM_LICENSE_DISPLAY_NAMES should produce the expected name."""
        for raw_name, expected_name in CUSTOM_LICENSE_DISPLAY_NAMES.items():
            metadata = {
                "cardData": {"license": "other", "license_name": raw_name},
                "siblings": [],
            }
            _, name, _ = resolve_license_info(metadata, "org/model")
            assert name == expected_name


# ===================================================================
# fetch_model integration — license fields populated via mock
# ===================================================================


def _mock_fetch_with_metadata(model_name, hf_id, config, metadata):
    """Call fetch_model with mocked config.json and API metadata."""
    with (
        patch("pipeline.sources.huggingface.fetch_hf_config", return_value=config),
        patch("pipeline.sources.huggingface.fetch_hf_metadata", return_value=metadata),
    ):
        return fetch_model(model_name, hf_id)


class TestFetchModelLicenseIntegration:
    """Integration tests: fetch_model populates license fields on ModelSpec."""

    def test_mit_license_from_metadata(self):
        metadata = {
            "cardData": {"license": "mit"},
            "siblings": [{"rfilename": "LICENSE"}],
        }
        spec = _mock_fetch_with_metadata(
            "DS", "deepseek-ai/DeepSeek-V3.2-Speciale", DEEPSEEK_V32_CONFIG, metadata
        )
        assert spec.license_spdx == "mit"
        assert spec.license_name == "MIT"
        assert "deepseek-ai/DeepSeek-V3.2-Speciale" in spec.license_url

    def test_apache_license_from_metadata(self):
        metadata = {
            "cardData": {
                "license": "apache-2.0",
                "license_link": "https://huggingface.co/Qwen/model/blob/main/LICENSE",
            },
            "siblings": [],
        }
        spec = _mock_fetch_with_metadata(
            "Q3", "Qwen/Qwen3-Coder-480B", QWEN3_CODER_CONFIG, metadata
        )
        assert spec.license_spdx == "apache-2.0"
        assert spec.license_name == "Apache 2.0"
        assert spec.license_url == "https://huggingface.co/Qwen/model/blob/main/LICENSE"

    def test_no_metadata_returns_none_license(self):
        """When HF API metadata fetch fails, license fields should be None."""
        spec = _mock_fetch_with_metadata(
            "GLM", "zai-org/GLM-4.7", GLM_47_CONFIG, None
        )
        assert spec.license_spdx is None
        assert spec.license_name is None
        assert spec.license_url is None

    def test_license_fields_coexist_with_existing_fields(self):
        """License fields should not break existing ModelSpec fields."""
        metadata = {
            "cardData": {"license": "mit"},
            "siblings": [],
        }
        spec = _mock_fetch_with_metadata(
            "DS", "deepseek-ai/DeepSeek-V3.2-Speciale", DEEPSEEK_V32_CONFIG, metadata
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
    "license_spdx",
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
            license_spdx="mit",
            license_name="MIT",
            license_url="https://example.com/LICENSE",
        )
        dumped = spec.model_dump()
        assert dumped["license_spdx"] == "mit"
        assert dumped["license_name"] == "MIT"
        assert dumped["license_url"] == "https://example.com/LICENSE"

    def test_modelspec_json_roundtrip_null_license(self):
        """License fields should be null when not set."""
        spec = ModelSpec(model_name="TestModel")
        dumped = spec.model_dump()
        assert dumped["license_spdx"] is None
        assert dumped["license_name"] is None
        assert dumped["license_url"] is None
