import { describe, it, expect } from "vitest";
import type { Model, Precision } from "@/types";
import {
  getModelMemory,
  getActiveModelMemory,
  gpusNeeded,
  calcKvCachePerToken,
  resolveModelPrecision,
  resolveKvPrecisionBytes,
  isNvLink,
  WEIGHT_OVERHEAD_FACTOR,
} from "../calculations";

// ---------------------------------------------------------------------------
// Test fixtures — models from models.json
// ---------------------------------------------------------------------------

const DEEPSEEK_V32: Model = {
  model_name: "DeepSeek-V3.2-Reasoner",
  learnable_params_b: 671.1,
  active_params_b: 37.7,
  architecture: "MoE",
  context_length: 163840,
  precision: "FP8",
  routed_expert_params_b: null,
  attention_type: "MLA",
  num_hidden_layers: 61,
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: null,
  head_dim: null,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  hf_model_id: null,
  model_url: null,
  kv_elems_per_token: null,
  license_name: null,
  license_url: null,
};

const GLM_47: Model = {
  model_name: "GLM-4.7",
  learnable_params_b: 352.8,
  active_params_b: 33.7,
  architecture: "MoE",
  context_length: 202752,
  precision: "BF16",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 92,
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: 8,
  head_dim: 128,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
  model_url: null,
  kv_elems_per_token: null,
  license_name: null,
  license_url: null,
};

const KIMI_K2: Model = {
  model_name: "Kimi-K2-Thinking",
  learnable_params_b: 1026.4,
  active_params_b: 32.9,
  architecture: "MoE",
  context_length: 262144,
  precision: "INT4",
  routed_expert_params_b: 1014.7,
  attention_type: "MLA",
  num_hidden_layers: 61,
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: null,
  head_dim: null,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  hf_model_id: null,
  model_url: null,
  kv_elems_per_token: null,
  license_name: null,
  license_url: null,
};

const QWEN3_CODER: Model = {
  model_name: "Qwen3-Coder-480B",
  learnable_params_b: 480.2,
  active_params_b: 35.5,
  architecture: "MoE",
  context_length: 262144,
  precision: "BF16",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 62,
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: 8,
  head_dim: 128,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
  model_url: null,
  kv_elems_per_token: null,
  license_name: null,
  license_url: null,
};

// DeepSeek-V4-Pro: mixed FP4-expert / FP8-rest checkpoint with the bespoke
// "DSV4" compressed + sparse-indexed KV cache (kv_elems_per_token precomputed
// by the pipeline: 31 ratio-128 layers × 512/128 + 30 ratio-4 layers ×
// (512/4 + 128/4) = 124 + 4800 = 4924).
const DEEPSEEK_V4: Model = {
  model_name: "DeepSeek-V4-Pro",
  learnable_params_b: 1598.8,
  active_params_b: 50.6,
  architecture: "MoE",
  context_length: 1048576,
  precision: "FP4",
  routed_expert_params_b: 1572.8,
  attention_type: "DSV4",
  num_hidden_layers: 61,
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: null,
  head_dim: null,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  kv_elems_per_token: 4924,
  hf_model_id: "deepseek-ai/DeepSeek-V4-Pro",
  model_url: null,
  license_name: "MIT",
  license_url: "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/LICENSE",
};

/** Model with null KV cache fields — should return 0 for all KV functions. */
const INCOMPLETE_MODEL: Model = {
  model_name: "Incomplete",
  learnable_params_b: 100,
  active_params_b: null,
  architecture: "Dense",
  context_length: 8192,
  precision: null,
  routed_expert_params_b: null,
  attention_type: null,
  num_hidden_layers: null,
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: null,
  head_dim: null,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
  model_url: null,
  kv_elems_per_token: null,
  license_name: null,
  license_url: null,
};

// ---------------------------------------------------------------------------
// getModelMemory
// ---------------------------------------------------------------------------

describe("getModelMemory", () => {
  it("returns params × bytes_per_param for each precision", () => {
    const cases: [Precision, number][] = [
      ["fp32", 352.8 * 4],
      ["fp16", 352.8 * 2],
      ["bf16", 352.8 * 2],
      ["fp8", 352.8 * 1],
      ["int8", 352.8 * 1],
      ["int4", 352.8 * 0.5625],
    ];
    for (const [prec, expected] of cases) {
      expect(getModelMemory(GLM_47, prec)).toBeCloseTo(expected, 1);
    }
  });

  it("FP8 model uses 1 byte per param", () => {
    // DeepSeek-V3.2 at FP8: 671.1 × 1 = 671.1 GB
    expect(getModelMemory(DEEPSEEK_V32, "fp8")).toBeCloseTo(671.1, 1);
  });

  it("mixed-precision: routed experts at INT4, rest at BF16", () => {
    // Kimi-K2: 1014.7B routed × 0.5625 + (1026.4 - 1014.7)B non-routed × 2.0
    // = 570.77 + 11.7 × 2.0 = 570.77 + 23.4 = 594.17 GB
    const mem = getModelMemory(KIMI_K2, "int4");
    expect(mem).toBeCloseTo(1014.7 * 0.5625 + (1026.4 - 1014.7) * 2.0, 1);
  });

  it("mixed-precision model at non-int4 uses standard formula", () => {
    // If someone overrides to fp16, all params use 2 bytes (no mixed-precision split)
    expect(getModelMemory(KIMI_K2, "fp16")).toBeCloseTo(1026.4 * 2, 1);
  });

  it("FP4-mixed (DeepSeek-V4): routed experts at FP4, rest at FP8 (not BF16)", () => {
    // 1572.8B routed × 0.5625 + (1598.8 - 1572.8)B non-routed × 1.0 (FP8)
    const mem = getModelMemory(DEEPSEEK_V4, "int4");
    expect(mem).toBeCloseTo(1572.8 * 0.5625 + (1598.8 - 1572.8) * 1.0, 1);

    // The FP8 split matters: legacy INT4-mixed would put the rest at BF16,
    // over-counting the ~26B non-expert params by ~26 GB.
    const legacyInt4 = { ...DEEPSEEK_V4, precision: "INT4" };
    expect(getModelMemory(legacyInt4, "int4")).toBeCloseTo(
      1572.8 * 0.5625 + (1598.8 - 1572.8) * 2.0,
      1,
    );
  });

  it("returns null when learnable_params_b is null", () => {
    const m = { ...GLM_47, learnable_params_b: null };
    expect(getModelMemory(m, "fp16")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// getActiveModelMemory — bytes read per decode step (MoE-aware)
// ---------------------------------------------------------------------------

describe("getActiveModelMemory", () => {
  it("MoE with uniform precision uses active_params_b", () => {
    // GLM-4.7: MoE, active_params_b=33.7, no routed split at BF16
    // active memory = 33.7 × 2 = 67.4 GB
    expect(getActiveModelMemory(GLM_47, "bf16")).toBeCloseTo(33.7 * 2, 1);
  });

  it("MoE at FP8 uses active_params_b", () => {
    // DeepSeek-V3.2: MoE, active_params_b=37.7, FP8
    // active memory = 37.7 × 1 = 37.7 GB
    expect(getActiveModelMemory(DEEPSEEK_V32, "fp8")).toBeCloseTo(37.7 * 1, 1);
  });

  it("MoE with INT4 mixed-precision splits active routed vs non-routed", () => {
    // Kimi-K2: active_params_b=32.9, routed=1014.7, learnable=1026.4
    // non_routed = 1026.4 - 1014.7 = 11.7B at BF16 (2 bytes)
    // active_routed = 32.9 - 11.7 = 21.2B at INT4 (0.5625 bytes)
    // active memory = 21.2 × 0.5625 + 11.7 × 2.0 = 11.925 + 23.4 = 35.325 GB
    const mem = getActiveModelMemory(KIMI_K2, "int4");
    expect(mem).toBeCloseTo(21.2 * 0.5625 + 11.7 * 2.0, 1);
  });

  it("FP4-mixed (DeepSeek-V4): active routed at FP4, non-routed at FP8", () => {
    // non_routed = 1598.8 - 1572.8 = 26.0B at FP8 (1.0 byte)
    // active_routed = 50.6 - 26.0 = 24.6B at FP4 (0.5625 bytes)
    const mem = getActiveModelMemory(DEEPSEEK_V4, "int4");
    const nonRouted = 1598.8 - 1572.8;
    const activeRouted = 50.6 - nonRouted;
    expect(mem).toBeCloseTo(activeRouted * 0.5625 + nonRouted * 1.0, 1);
  });

  it("dense model falls back to getModelMemory (all params active)", () => {
    const dense: Model = {
      ...INCOMPLETE_MODEL,
      learnable_params_b: 70,
      active_params_b: 70,
      architecture: "Dense",
    };
    // 70 × 2 = 140 GB at fp16
    expect(getActiveModelMemory(dense, "fp16")).toBeCloseTo(70 * 2, 1);
    expect(getActiveModelMemory(dense, "fp16")).toBe(getModelMemory(dense, "fp16"));
  });

  it("null active_params_b falls back to getModelMemory", () => {
    expect(getActiveModelMemory(INCOMPLETE_MODEL, "fp16")).toBe(
      getModelMemory(INCOMPLETE_MODEL, "fp16"),
    );
  });

  it("active memory is much smaller than total for MoE models", () => {
    // Kimi-K2: total = 594 GB, active = 35 GB → ~17x smaller
    const total = getModelMemory(KIMI_K2, "int4")!;
    const active = getActiveModelMemory(KIMI_K2, "int4")!;
    expect(total / active).toBeGreaterThan(10);
  });
});

// ---------------------------------------------------------------------------
// resolveModelPrecision — maps model metadata to Precision type
// ---------------------------------------------------------------------------

describe("resolveModelPrecision", () => {
  it("resolves FP8 models", () => {
    expect(resolveModelPrecision(DEEPSEEK_V32)).toBe("fp8");
  });

  it("resolves BF16 models", () => {
    expect(resolveModelPrecision(GLM_47)).toBe("bf16");
    expect(resolveModelPrecision(QWEN3_CODER)).toBe("bf16");
  });

  it("resolves MXFP8 to fp8 (8-bit format, 1 byte/param)", () => {
    // MiniMax-M3-MXFP8: 426.3B at MXFP8 must size as 8-bit (~426 GB), not
    // fall back to fp16 (~853 GB) which would wrongly need 13 H100s vs 7.
    const m = { ...DEEPSEEK_V32, precision: "MXFP8" };
    expect(resolveModelPrecision(m)).toBe("fp8");
  });

  it("resolves INT4 to int4", () => {
    expect(resolveModelPrecision(KIMI_K2)).toBe("int4");
  });

  it("resolves INT4-mixed to int4 (legacy)", () => {
    const m = { ...KIMI_K2, precision: "INT4-mixed" };
    expect(resolveModelPrecision(m)).toBe("int4");
  });

  it("falls back to fp16 for null precision", () => {
    expect(resolveModelPrecision(INCOMPLETE_MODEL)).toBe("fp16");
  });

  it("falls back to fp16 for unknown precision strings", () => {
    const m = { ...GLM_47, precision: "WEIRD" };
    expect(resolveModelPrecision(m)).toBe("fp16");
  });

  it("resolves FLOAT4 (modelopt FP4) to int4", () => {
    const m = { ...KIMI_K2, precision: "FLOAT4" };
    expect(resolveModelPrecision(m)).toBe("int4");
  });

  it("resolves FP4 (DeepSeek-V4 mixed) to int4", () => {
    expect(resolveModelPrecision(DEEPSEEK_V4)).toBe("int4");
  });

  it("resolves FLOAT8 (modelopt FP8) to fp8", () => {
    const m = { ...DEEPSEEK_V32, precision: "FLOAT8" };
    expect(resolveModelPrecision(m)).toBe("fp8");
  });
});

// ---------------------------------------------------------------------------
// gpusNeeded
// ---------------------------------------------------------------------------

describe("gpusNeeded", () => {
  it("rounds up to next GPU", () => {
    expect(gpusNeeded(80, 80)).toBe(1);
    expect(gpusNeeded(81, 80)).toBe(2);
    expect(gpusNeeded(640, 80)).toBe(8);
    expect(gpusNeeded(641, 80)).toBe(9);
  });

  it("accounts for overhead factor correctly", () => {
    // 352.8B at FP16 = 705.6 GB weight
    // With 1.15× overhead = 811.44 GB
    // H100 80GB → ceil(811.44/80) = 11
    const memGb = 352.8 * 2;
    expect(gpusNeeded(memGb * WEIGHT_OVERHEAD_FACTOR, 80)).toBe(11);
  });
});

// ---------------------------------------------------------------------------
// calcKvCachePerToken — architecture-aware formulas
// ---------------------------------------------------------------------------

describe("calcKvCachePerToken", () => {
  it("MLA: layers × (kv_lora_rank + qk_rope_head_dim) × 2 bytes (FP16 default)", () => {
    // DeepSeek-V3.2: 61 × (512 + 64) × 2 = 70,272 bytes
    const kvGb = calcKvCachePerToken(DEEPSEEK_V32);
    const kvBytes = kvGb * 1024 * 1024 * 1024;
    expect(kvBytes).toBeCloseTo(70_272, 0);
  });

  it("GQA: 2 × layers × num_kv_heads × head_dim × 2 bytes (FP16 default)", () => {
    // GLM-4.7: 2 × 92 × 8 × 128 × 2 = 376,832 bytes
    const kvGb = calcKvCachePerToken(GLM_47);
    const kvBytes = kvGb * 1024 * 1024 * 1024;
    expect(kvBytes).toBeCloseTo(376_832, 0);
  });

  it("FP8 KV cache is exactly half of FP16", () => {
    const fp16 = calcKvCachePerToken(DEEPSEEK_V32, 2);
    const fp8 = calcKvCachePerToken(DEEPSEEK_V32, 1);
    expect(fp8).toBeCloseTo(fp16 / 2, 12);

    const fp16gqa = calcKvCachePerToken(GLM_47, 2);
    const fp8gqa = calcKvCachePerToken(GLM_47, 1);
    expect(fp8gqa).toBeCloseTo(fp16gqa / 2, 12);
  });

  it("MLA is ~5× more KV-efficient than GQA", () => {
    const mlaBytes = calcKvCachePerToken(DEEPSEEK_V32);
    const gqaBytes = calcKvCachePerToken(GLM_47);
    const ratio = gqaBytes / mlaBytes;
    expect(ratio).toBeGreaterThan(4);
    expect(ratio).toBeLessThan(7);
  });

  it("Kimi-K2 MLA matches DeepSeek MLA (same architecture dims)", () => {
    expect(calcKvCachePerToken(KIMI_K2)).toBe(calcKvCachePerToken(DEEPSEEK_V32));
  });

  it("Qwen3 GQA: 2 × 62 × 8 × 128 × 2 = 253,952 bytes", () => {
    const kvGb = calcKvCachePerToken(QWEN3_CODER);
    const kvBytes = kvGb * 1024 * 1024 * 1024;
    expect(kvBytes).toBeCloseTo(253_952, 0);
  });

  it("DSV4: kv_elems_per_token × 2 bytes (FP16 default)", () => {
    // DeepSeek-V4: 4924 × 2 = 9848 bytes/token
    const kvBytes = calcKvCachePerToken(DEEPSEEK_V4) * 1024 * 1024 * 1024;
    expect(kvBytes).toBeCloseTo(9_848, 0);
  });

  it("DSV4 FP8 KV cache is exactly half of FP16", () => {
    expect(calcKvCachePerToken(DEEPSEEK_V4, 1)).toBeCloseTo(
      calcKvCachePerToken(DEEPSEEK_V4, 2) / 2,
      12,
    );
  });

  it("DSV4 is far more KV-efficient than the equivalent dense-MLA baseline", () => {
    // V4's compressed KV (~9.8 KB/token) is a small fraction of a 61-layer
    // MLA model's (~70 KB/token) — the headline long-context win.
    expect(calcKvCachePerToken(DEEPSEEK_V4)).toBeLessThan(
      calcKvCachePerToken(DEEPSEEK_V32) / 5,
    );
  });

  it("returns 0 if DSV4 kv_elems_per_token is null", () => {
    const broken = { ...DEEPSEEK_V4, kv_elems_per_token: null };
    expect(calcKvCachePerToken(broken)).toBe(0);
  });

  it("returns 0 for unknown attention_type", () => {
    expect(calcKvCachePerToken(INCOMPLETE_MODEL)).toBe(0);
  });

  it("returns 0 if MLA fields are null", () => {
    const broken = { ...DEEPSEEK_V32, kv_lora_rank: null };
    expect(calcKvCachePerToken(broken)).toBe(0);
  });

  it("returns 0 if GQA fields are null", () => {
    const broken = { ...GLM_47, num_kv_heads: null };
    expect(calcKvCachePerToken(broken)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// resolveKvPrecisionBytes
// ---------------------------------------------------------------------------

describe("resolveKvPrecisionBytes", () => {
  it("fp16 always returns 2", () => {
    expect(resolveKvPrecisionBytes("fp16", "H100")).toBe(2);
    expect(resolveKvPrecisionBytes("fp16", "A100")).toBe(2);
    expect(resolveKvPrecisionBytes("fp16", "V100")).toBe(2);
  });

  it("fp8 always returns 1", () => {
    expect(resolveKvPrecisionBytes("fp8", "H100")).toBe(1);
    expect(resolveKvPrecisionBytes("fp8", "A100")).toBe(1);
    expect(resolveKvPrecisionBytes("fp8", "V100")).toBe(1);
  });

  it("auto returns 1 for Hopper GPUs", () => {
    expect(resolveKvPrecisionBytes("auto", "H100")).toBe(1);
    expect(resolveKvPrecisionBytes("auto", "H200")).toBe(1);
  });

  it("auto returns 1 for Blackwell GPUs", () => {
    expect(resolveKvPrecisionBytes("auto", "B200")).toBe(1);
    expect(resolveKvPrecisionBytes("auto", "B300")).toBe(1);
  });

  it("auto returns 1 for Ada Lovelace GPUs", () => {
    expect(resolveKvPrecisionBytes("auto", "L4")).toBe(1);
    expect(resolveKvPrecisionBytes("auto", "L40S")).toBe(1);
    expect(resolveKvPrecisionBytes("auto", "RTX4090")).toBe(1);
  });

  it("auto returns 2 for Ampere GPUs", () => {
    expect(resolveKvPrecisionBytes("auto", "A100")).toBe(2);
    expect(resolveKvPrecisionBytes("auto", "A100_80G")).toBe(2);
  });

  it("auto returns 2 for Volta GPUs", () => {
    expect(resolveKvPrecisionBytes("auto", "V100")).toBe(2);
  });

  it("auto returns 2 for unknown GPUs", () => {
    expect(resolveKvPrecisionBytes("auto", "FakeGPU")).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// isNvLink
// ---------------------------------------------------------------------------

describe("isNvLink", () => {
  it("returns true for NVLink variants and NVLink-based tiers", () => {
    expect(isNvLink("NVLink sxm4")).toBe(true);
    expect(isNvLink("NVLink sxm5")).toBe(true);
    expect(isNvLink("nvlink")).toBe(true);
    expect(isNvLink("NVLINK")).toBe(true);
    // Tier enums are NVLink-based fabrics too.
    expect(isNvLink("nvlink_paired")).toBe(true);
    expect(isNvLink("nvswitch")).toBe(true);
  });

  it("returns false for non-NVLink", () => {
    expect(isNvLink("PCIe")).toBe(false);
    expect(isNvLink("pcie")).toBe(false);
    expect(isNvLink("none")).toBe(false);
    expect(isNvLink(null)).toBe(false);
    expect(isNvLink("")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// WEIGHT_OVERHEAD_FACTOR sanity
// ---------------------------------------------------------------------------

describe("WEIGHT_OVERHEAD_FACTOR", () => {
  it("is 1.15", () => {
    expect(WEIGHT_OVERHEAD_FACTOR).toBe(1.15);
  });
});
