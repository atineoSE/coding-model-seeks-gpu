"""Tests for KV cache field extraction in fetch_model().

Uses unittest.mock to avoid real HTTP calls.  The embedded configs are
copied from test_param_counter.py and represent real HuggingFace configs.
"""

from unittest.mock import patch

import pytest

from pipeline.sources.huggingface import fetch_model

# Re-use the embedded configs from test_param_counter.  Import them rather
# than duplicating to keep a single source of truth.
from tests.test_param_counter import (
    DEEPSEEK_V32_CONFIG,
    GLM_47_CONFIG,
    KIMI_K2_THINKING_CONFIG,
    KIMI_K25_CONFIG,
    MINIMAX_M25_CONFIG,
    NEMOTRON_H_30B_CONFIG,
    QWEN3_CODER_CONFIG,
)


def _mock_fetch(model_name: str, hf_id: str, config: dict):
    """Call fetch_model with mocked HTTP, returning a ModelSpec."""
    with (
        patch("pipeline.sources.huggingface.fetch_hf_config", return_value=config),
        patch("pipeline.sources.huggingface.fetch_hf_param_count", return_value=None),
    ):
        return fetch_model(model_name, hf_id)


# ===================================================================
# MLA models — DeepSeek, Kimi-K2, Kimi-K2.5
# ===================================================================


class TestMlaKvFields:
    """MLA models should have kv_lora_rank, qk_rope_head_dim, no GQA fields."""

    def test_deepseek_v32(self):
        spec = _mock_fetch("DS", "deepseek-ai/DeepSeek-V3.2-Speciale", DEEPSEEK_V32_CONFIG)
        assert spec.attention_type == "MLA"
        assert spec.num_hidden_layers == 61
        assert spec.kv_lora_rank == 512
        assert spec.qk_rope_head_dim == 64
        assert spec.num_kv_heads is None
        assert spec.head_dim is None

    def test_kimi_k2_thinking(self):
        spec = _mock_fetch("K2", "moonshotai/Kimi-K2-Thinking", KIMI_K2_THINKING_CONFIG)
        assert spec.attention_type == "MLA"
        assert spec.num_hidden_layers == 61
        assert spec.kv_lora_rank == 512
        assert spec.qk_rope_head_dim == 64

    def test_kimi_k25_multimodal_unwrap(self):
        """Kimi-K2.5 wraps text config in text_config — KV fields should still resolve."""
        spec = _mock_fetch("K25", "moonshotai/Kimi-K2.5", KIMI_K25_CONFIG)
        assert spec.attention_type == "MLA"
        assert spec.num_hidden_layers == 61
        assert spec.kv_lora_rank == 512
        assert spec.qk_rope_head_dim == 64


# ===================================================================
# GQA models — GLM-4.7, MiniMax-M2.5, Qwen3-Coder
# ===================================================================


class TestGqaKvFields:
    """GQA models should have num_kv_heads, head_dim, no MLA fields."""

    def test_glm_47(self):
        spec = _mock_fetch("GLM", "zai-org/GLM-4.7", GLM_47_CONFIG)
        assert spec.attention_type == "GQA"
        assert spec.num_hidden_layers == 92
        assert spec.num_kv_heads == 8
        assert spec.head_dim == 128
        assert spec.kv_lora_rank is None
        assert spec.qk_rope_head_dim is None

    def test_minimax_m25(self):
        spec = _mock_fetch("MM", "MiniMaxAI/MiniMax-M2.5", MINIMAX_M25_CONFIG)
        assert spec.attention_type == "GQA"
        assert spec.num_hidden_layers == 62
        assert spec.num_kv_heads == 8
        assert spec.head_dim == 128

    def test_qwen3_coder(self):
        spec = _mock_fetch("Q3", "Qwen/Qwen3-Coder-480B", QWEN3_CODER_CONFIG)
        assert spec.attention_type == "GQA"
        assert spec.num_hidden_layers == 62
        assert spec.num_kv_heads == 8
        assert spec.head_dim == 128

    def test_head_dim_fallback_from_hidden_and_heads(self):
        """When head_dim is not explicit, compute from hidden_size / num_attention_heads."""
        config = {**GLM_47_CONFIG}
        del config["head_dim"]  # Force fallback: 5120 // 96 ≈ 53
        spec = _mock_fetch("GLM-no-head-dim", "fake/id", config)
        assert spec.head_dim == 5120 // 96


# ===================================================================
# KV cache spot-check values (bytes per token)
# ===================================================================


class TestKvCacheValues:
    """Verify that the extracted fields produce correct KV cache bytes/token."""

    @staticmethod
    def _kv_bytes_per_token(spec) -> int:
        """Replicate the frontend formula to cross-check pipeline data."""
        if spec.attention_type == "MLA":
            return spec.num_hidden_layers * (spec.kv_lora_rank + spec.qk_rope_head_dim) * 2
        elif spec.attention_type == "GQA":
            return 2 * spec.num_hidden_layers * spec.num_kv_heads * spec.head_dim * 2
        raise ValueError(f"Unknown attention_type: {spec.attention_type}")

    def test_deepseek_mla_bytes(self):
        spec = _mock_fetch("DS", "deepseek-ai/DS", DEEPSEEK_V32_CONFIG)
        assert self._kv_bytes_per_token(spec) == 70_272  # 61 × 576 × 2

    def test_glm_gqa_bytes(self):
        spec = _mock_fetch("GLM", "zai-org/GLM", GLM_47_CONFIG)
        assert self._kv_bytes_per_token(spec) == 376_832  # 2 × 92 × 8 × 128 × 2

    def test_qwen3_gqa_bytes(self):
        spec = _mock_fetch("Q3", "Qwen/Q3", QWEN3_CODER_CONFIG)
        assert self._kv_bytes_per_token(spec) == 253_952  # 2 × 62 × 8 × 128 × 2

    def test_mla_is_more_efficient_than_gqa(self):
        ds = _mock_fetch("DS", "ds/id", DEEPSEEK_V32_CONFIG)
        glm = _mock_fetch("GLM", "glm/id", GLM_47_CONFIG)
        ratio = self._kv_bytes_per_token(glm) / self._kv_bytes_per_token(ds)
        assert 4 < ratio < 7  # MLA ~5× more efficient


# ===================================================================
# Error paths — missing fields should raise
# ===================================================================


class TestKvFieldErrors:
    """Pipeline should raise ValueError when required KV cache fields are missing."""

    def test_missing_num_hidden_layers(self):
        config = {**DEEPSEEK_V32_CONFIG}
        del config["num_hidden_layers"]
        # count_params_from_config also requires this field, so the error
        # may come from either the param counter or our KV extraction.
        with pytest.raises(ValueError, match="num_hidden_layers"):
            _mock_fetch("DS", "ds/id", config)

    def test_missing_kv_lora_rank_mla(self):
        config = {**DEEPSEEK_V32_CONFIG}
        del config["kv_lora_rank"]
        # count_params_from_config also needs kv_lora_rank, so it may raise first.
        # Either way, a ValueError should propagate.
        with pytest.raises(ValueError):
            _mock_fetch("DS", "ds/id", config)

    def test_missing_qk_rope_head_dim_mla(self):
        config = {**DEEPSEEK_V32_CONFIG}
        del config["qk_rope_head_dim"]
        with pytest.raises(ValueError):
            _mock_fetch("DS", "ds/id", config)

    def test_missing_num_key_value_heads_gqa(self):
        config = {**GLM_47_CONFIG}
        del config["num_key_value_heads"]
        with pytest.raises(ValueError):
            _mock_fetch("GLM", "glm/id", config)

    def test_missing_head_dim_and_hidden_size_gqa(self):
        """When both head_dim and hidden_size are missing, should raise."""
        config = {**GLM_47_CONFIG}
        del config["head_dim"]
        del config["hidden_size"]
        with pytest.raises(ValueError):
            _mock_fetch("GLM", "glm/id", config)

    def test_unknown_model_type_raises(self):
        from pipeline.errors import UnsupportedArchitecture

        config = {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32}
        with pytest.raises(UnsupportedArchitecture, match="model_type='llama'"):
            _mock_fetch("Llama", "meta/llama", config)


# ===================================================================
# Hybrid (nemotron_h) — KV cache fields
# ===================================================================


class TestHybridKvFields:
    """Hybrid models should have num_kv_layers set to the attention layer count."""

    def test_nemotron_h_attention_type(self):
        spec = _mock_fetch("Nem", "nvidia/nem", NEMOTRON_H_30B_CONFIG)
        assert spec.attention_type == "GQA"

    def test_nemotron_h_num_hidden_layers(self):
        spec = _mock_fetch("Nem", "nvidia/nem", NEMOTRON_H_30B_CONFIG)
        assert spec.num_hidden_layers == 52

    def test_nemotron_h_num_kv_layers(self):
        spec = _mock_fetch("Nem", "nvidia/nem", NEMOTRON_H_30B_CONFIG)
        assert spec.num_kv_layers == 6

    def test_nemotron_h_num_kv_heads(self):
        spec = _mock_fetch("Nem", "nvidia/nem", NEMOTRON_H_30B_CONFIG)
        assert spec.num_kv_heads == 2

    def test_nemotron_h_head_dim(self):
        spec = _mock_fetch("Nem", "nvidia/nem", NEMOTRON_H_30B_CONFIG)
        assert spec.head_dim == 128

    def test_nemotron_h_kv_bytes_per_token(self):
        """KV bytes = 2 (K+V) × 6 layers × 2 heads × 128 dim × 2 bytes = 6,144."""
        spec = _mock_fetch("Nem", "nvidia/nem", NEMOTRON_H_30B_CONFIG)
        kv_bytes = 2 * spec.num_kv_layers * spec.num_kv_heads * spec.head_dim * 2
        assert kv_bytes == 6_144

    def test_existing_models_num_kv_layers_none(self):
        """Existing (non-hybrid) models should have num_kv_layers=None."""
        for name, hf_id, config in [
            ("DS", "deepseek-ai/DS", DEEPSEEK_V32_CONFIG),
            ("GLM", "zai-org/GLM", GLM_47_CONFIG),
            ("Q3", "Qwen/Q3", QWEN3_CODER_CONFIG),
            ("MM", "MiniMaxAI/MM", MINIMAX_M25_CONFIG),
        ]:
            spec = _mock_fetch(name, hf_id, config)
            assert spec.num_kv_layers is None, f"{name} should have num_kv_layers=None"
