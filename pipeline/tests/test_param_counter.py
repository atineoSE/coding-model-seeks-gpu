"""Tests for config-based parameter counting and precision detection.

All configs are embedded — no HTTP calls needed.
"""

import pytest

from pipeline.sources.param_counter import (
    ParamCountResult,
    count_params_from_config,
    detect_precision,
)

# ---------------------------------------------------------------------------
# Minimal embedded configs (only the fields param_counter needs)
# ---------------------------------------------------------------------------

DEEPSEEK_V32_CONFIG = {
    "model_type": "deepseek_v32",
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "vocab_size": 129280,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "first_k_dense_replace": 3,
    "num_nextn_predict_layers": 1,
    "tie_word_embeddings": False,
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
    },
    "torch_dtype": "bfloat16",
}

GLM_47_CONFIG = {
    "model_type": "glm4_moe",
    "hidden_size": 5120,
    "intermediate_size": 12288,
    "num_hidden_layers": 92,
    "num_attention_heads": 96,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151552,
    "n_routed_experts": 160,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 1536,
    "first_k_dense_replace": 3,
    "num_nextn_predict_layers": 1,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}

# Kimi-K2.5 is multimodal — config wraps text backbone in text_config
KIMI_K25_CONFIG = {
    "model_type": "kimi_k25",
    "text_config": {
        "model_type": "kimi_k2",
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "num_hidden_layers": 61,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "vocab_size": 163840,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 384,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "first_k_dense_replace": 1,
        "num_nextn_predict_layers": 0,
        "tie_word_embeddings": False,
        "max_position_embeddings": 262144,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int"},
                }
            },
            "ignore": [
                "lm_head",
                "re:.*self_attn.*",
                "re:.*shared_experts.*",
            ],
        },
    },
}

KIMI_K2_THINKING_CONFIG = {
    "model_type": "kimi_k2",
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "num_hidden_layers": 61,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "vocab_size": 163840,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "first_k_dense_replace": 1,
    "num_nextn_predict_layers": 0,
    "tie_word_embeddings": False,
    "quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "int"},
            }
        },
        "ignore": [
            "lm_head",
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
        ],
    },
    "torch_dtype": "bfloat16",
}

MINIMAX_M25_CONFIG = {
    "model_type": "minimax_m2",
    "hidden_size": 3072,
    "intermediate_size": 1536,
    "num_hidden_layers": 62,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 200064,
    "num_local_experts": 256,
    "num_experts_per_tok": 8,
    "shared_intermediate_size": 0,
    "num_mtp_modules": 3,
    "tie_word_embeddings": False,
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "float8_e4m3fn",
    },
}

# M2.1 has identical architecture to M2.5
MINIMAX_M21_CONFIG = MINIMAX_M25_CONFIG.copy()

QWEN3_CODER_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 6144,
    "intermediate_size": 8192,
    "num_hidden_layers": 62,
    "num_attention_heads": 96,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "num_experts": 160,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2560,
    "shared_expert_intermediate_size": 0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


# ===================================================================
# Ground truth validation
# ===================================================================

MINIMAX_SAFETENSORS_TRUTH = 228_703_644_928  # from HF API


class TestGroundTruth:
    """Validate param counts against safetensors ground truth."""

    def test_minimax_m25_total_params(self):
        result = count_params_from_config(MINIMAX_M25_CONFIG)
        pct_error = abs(result.total_params - MINIMAX_SAFETENSORS_TRUTH) / MINIMAX_SAFETENSORS_TRUTH
        assert pct_error < 0.01, (
            f"MiniMax-M2.5 total_params={result.total_params:,} vs "
            f"truth={MINIMAX_SAFETENSORS_TRUTH:,}, error={pct_error:.4%}"
        )

    def test_minimax_m21_same_as_m25(self):
        r1 = count_params_from_config(MINIMAX_M25_CONFIG)
        r2 = count_params_from_config(MINIMAX_M21_CONFIG)
        assert r1.total_params == r2.total_params


# ===================================================================
# Cross-check: all 7 model configs produce reasonable values
# ===================================================================


class TestAllModels:
    """Verify all 7 models produce reasonable parameter counts."""

    @pytest.mark.parametrize(
        "name,config,expected_total_b,expected_active_b",
        [
            ("DeepSeek-V3.2", DEEPSEEK_V32_CONFIG, 671, 37),
            ("GLM-4.7", GLM_47_CONFIG, 353, 34),
            ("Kimi-K2.5", KIMI_K25_CONFIG, 1026, 33),
            ("Kimi-K2-Thinking", KIMI_K2_THINKING_CONFIG, 1026, 33),
            ("MiniMax-M2.5", MINIMAX_M25_CONFIG, 229, 11),
            ("MiniMax-M2.1", MINIMAX_M21_CONFIG, 229, 11),
            ("Qwen3-Coder-480B", QWEN3_CODER_CONFIG, 480, 35),
        ],
    )
    def test_reasonable_total(self, name, config, expected_total_b, expected_active_b):
        result = count_params_from_config(config)
        total_b = result.total_params / 1e9
        active_b = result.active_params / 1e9
        # Within 5% of expected (rough sanity check)
        assert (
            abs(total_b - expected_total_b) / expected_total_b < 0.05
        ), f"{name}: total={total_b:.1f}B, expected ~{expected_total_b}B"
        assert (
            abs(active_b - expected_active_b) / expected_active_b < 0.10
        ), f"{name}: active={active_b:.1f}B, expected ~{expected_active_b}B"

    def test_all_are_moe(self):
        for config in [
            DEEPSEEK_V32_CONFIG,
            GLM_47_CONFIG,
            KIMI_K25_CONFIG,
            KIMI_K2_THINKING_CONFIG,
            MINIMAX_M25_CONFIG,
            QWEN3_CODER_CONFIG,
        ]:
            result = count_params_from_config(config)
            assert result.num_moe_layers > 0, f"{result.model_type} should be MoE"

    def test_result_fields(self):
        result = count_params_from_config(DEEPSEEK_V32_CONFIG)
        assert isinstance(result, ParamCountResult)
        assert result.model_type == "deepseek_v32"
        assert result.num_layers == 61
        assert result.num_dense_layers == 3
        assert result.num_moe_layers == 58
        assert result.active_params < result.total_params
        assert result.routed_expert_params > 0
        assert result.routed_expert_params < result.total_params

    def test_routed_expert_params_dominates(self):
        """Routed experts should be the bulk of params in large MoE models."""
        for config in [KIMI_K2_THINKING_CONFIG, QWEN3_CODER_CONFIG]:
            result = count_params_from_config(config)
            fraction = result.routed_expert_params / result.total_params
            assert fraction > 0.9, (
                f"{result.model_type}: routed experts are only " f"{fraction:.1%} of total params"
            )


# ===================================================================
# Error path tests
# ===================================================================


class TestErrorPaths:
    def test_unknown_model_type(self):
        config = {"model_type": "llama", "hidden_size": 4096}
        with pytest.raises(ValueError, match="Unknown model_type='llama'"):
            count_params_from_config(config)

    def test_missing_required_field(self):
        config = {
            "model_type": "deepseek_v32",
            # hidden_size is missing
            "intermediate_size": 18432,
            "num_hidden_layers": 61,
        }
        with pytest.raises(ValueError, match="Required config field 'hidden_size' is missing"):
            count_params_from_config(config)

    def test_multimodal_without_text_config(self):
        config = {"model_type": "kimi_k25"}
        with pytest.raises(ValueError, match="requires 'text_config' dict"):
            count_params_from_config(config)

    def test_multimodal_text_config_wrong_type(self):
        config = {"model_type": "kimi_k25", "text_config": "not_a_dict"}
        with pytest.raises(ValueError, match="requires 'text_config' dict"):
            count_params_from_config(config)

    def test_unknown_quant_method(self):
        config = {
            "model_type": "qwen3_moe",
            **{k: v for k, v in QWEN3_CODER_CONFIG.items() if k != "torch_dtype"},
            "quantization_config": {"quant_method": "gptq"},
        }
        with pytest.raises(ValueError, match="Unknown quant_method='gptq'"):
            detect_precision(config)

    def test_unknown_torch_dtype(self):
        config = {"model_type": "qwen3_moe", "torch_dtype": "float8"}
        with pytest.raises(ValueError, match="Unknown torch_dtype='float8'"):
            detect_precision(config)


# ===================================================================
# Precision detection tests
# ===================================================================


class TestPrecision:
    def test_deepseek_fp8(self):
        info = detect_precision(DEEPSEEK_V32_CONFIG)
        assert info.dtype_str == "FP8"
        assert info.bytes_per_param == 1.0
        assert not info.is_mixed

    def test_glm_bf16(self):
        info = detect_precision(GLM_47_CONFIG)
        assert info.dtype_str == "BF16"
        assert info.bytes_per_param == 2.0
        assert not info.is_mixed

    def test_kimi_k2_thinking_int4_mixed(self):
        info = detect_precision(KIMI_K2_THINKING_CONFIG)
        assert info.dtype_str == "INT4"
        assert info.bytes_per_param == 0.5
        assert info.is_mixed

    def test_kimi_k25_wrapped_int4_mixed(self):
        info = detect_precision(KIMI_K25_CONFIG)
        assert info.dtype_str == "INT4"
        assert info.bytes_per_param == 0.5
        assert info.is_mixed

    def test_qwen_bf16(self):
        info = detect_precision(QWEN3_CODER_CONFIG)
        assert info.dtype_str == "BF16"
        assert info.bytes_per_param == 2.0

    def test_minimax_fp8(self):
        info = detect_precision(MINIMAX_M25_CONFIG)
        assert info.dtype_str == "FP8"
        assert info.bytes_per_param == 1.0
