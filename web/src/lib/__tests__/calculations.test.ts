import { describe, it, expect } from "vitest";
import type { Model, GpuOffering, Precision } from "@/types";
import {
  getModelMemory,
  getActiveModelMemory,
  gpusNeeded,
  calcKvCachePerToken,
  calcKvCachePerRequest,
  calcDecodeThroughput,
  calcMaxConcurrentRequests,
  calcTeamCapacity,
  resolveModelPrecision,
  resolveKvPrecisionBytes,
  isNvLink,
  calcParallelismTopology,
  calcTpEfficiency,
  calcPpBubbleEfficiency,
  WEIGHT_OVERHEAD_FACTOR,
} from "../calculations";

// ---------------------------------------------------------------------------
// Test fixtures — models from models.json
// ---------------------------------------------------------------------------

const DEEPSEEK_V32: Model = {
  model_name: "DeepSeek-V3.2-Reasoner",
  published_param_count_b: null,
  learnable_params_b: 671.1,
  active_params_b: 37.7,
  architecture: "MoE",
  context_length: 163840,
  precision: "FP8",
  routed_expert_params_b: null,
  attention_type: "MLA",
  num_hidden_layers: 61,
  num_kv_heads: null,
  head_dim: null,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  hf_model_id: null,
};

const GLM_47: Model = {
  model_name: "GLM-4.7",
  published_param_count_b: null,
  learnable_params_b: 352.8,
  active_params_b: 33.7,
  architecture: "MoE",
  context_length: 202752,
  precision: "BF16",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 92,
  num_kv_heads: 8,
  head_dim: 128,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
};

const KIMI_K2: Model = {
  model_name: "Kimi-K2-Thinking",
  published_param_count_b: null,
  learnable_params_b: 1026.4,
  active_params_b: 32.9,
  architecture: "MoE",
  context_length: 262144,
  precision: "INT4",
  routed_expert_params_b: 1014.7,
  attention_type: "MLA",
  num_hidden_layers: 61,
  num_kv_heads: null,
  head_dim: null,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  hf_model_id: null,
};

const QWEN3_CODER: Model = {
  model_name: "Qwen3-Coder-480B",
  published_param_count_b: null,
  learnable_params_b: 480.2,
  active_params_b: 35.5,
  architecture: "MoE",
  context_length: 262144,
  precision: "BF16",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 62,
  num_kv_heads: 8,
  head_dim: 128,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
};

/** Model with null KV cache fields — should return 0 for all KV functions. */
const INCOMPLETE_MODEL: Model = {
  model_name: "Incomplete",
  published_param_count_b: null,
  learnable_params_b: 100,
  active_params_b: null,
  architecture: "Dense",
  context_length: 8192,
  precision: null,
  routed_expert_params_b: null,
  attention_type: null,
  num_hidden_layers: null,
  num_kv_heads: null,
  head_dim: null,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
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
// calcKvCachePerRequest
// ---------------------------------------------------------------------------

describe("calcKvCachePerRequest", () => {
  it("scales linearly with total tokens", () => {
    const kv1 = calcKvCachePerRequest(DEEPSEEK_V32, 1000, 500);
    const kv2 = calcKvCachePerRequest(DEEPSEEK_V32, 2000, 1000);
    expect(kv2).toBeCloseTo(kv1 * 2, 10);
  });

  it("DeepSeek MLA 1500 tokens ≈ 0.098 GB", () => {
    // 70,272 bytes/token × 1500 = 105,408,000 bytes ≈ 0.0982 GB
    const kv = calcKvCachePerRequest(DEEPSEEK_V32, 1000, 500);
    expect(kv).toBeCloseTo(0.0982, 3);
  });

  it("GLM GQA 1500 tokens ≈ 0.526 GB", () => {
    // 376,832 bytes/token × 1500 = 565,248,000 bytes ≈ 0.5264 GB
    const kv = calcKvCachePerRequest(GLM_47, 1000, 500);
    expect(kv).toBeCloseTo(0.5264, 3);
  });

  it("returns 0 for incomplete model", () => {
    expect(calcKvCachePerRequest(INCOMPLETE_MODEL, 1000, 500)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// calcMaxConcurrentRequests — physics-based VRAM budgeting
// ---------------------------------------------------------------------------

describe("calcMaxConcurrentRequests", () => {
  it("uses leftover VRAM after weights × overhead", () => {
    // GLM-4.7 at fp16: 352.8 × 2 = 705.6 GB weight
    // On 12 × H100 (960 GB total): kvBudget = 960 - 705.6 × 1.15 = 960 - 811.44 = 148.56 GB
    // KV per request (1000+500 tokens): 0.5264 GB
    // Max concurrent: floor(148.56 / 0.5264) = 282
    const result = calcMaxConcurrentRequests(GLM_47, "fp16", 960, 1000, 500);
    expect(result).toBe(282);
  });

  it("returns 0 when weights exceed VRAM", () => {
    // GLM-4.7 at fp16: 705.6 GB, with overhead 811.44 GB > 640 GB
    const result = calcMaxConcurrentRequests(GLM_47, "fp16", 640, 1000, 500);
    expect(result).toBe(0);
  });

  it("MLA models fit more concurrent requests than GQA", () => {
    // Both on 960 GB VRAM, same token counts
    // DeepSeek FP16: 1342.2 GB weight — doesn't fit at fp16
    // Use int8: 671.1 GB weight, overhead = 771.77 GB
    // kvBudget = 960 - 771.77 = 188.23 GB, kvPerReq = 0.0982 GB
    // Max = floor(188.23 / 0.0982) = 1917
    const dsResult = calcMaxConcurrentRequests(DEEPSEEK_V32, "int8", 960, 1000, 500);

    // GLM-4.7 at fp16: 282 (from test above)
    const glmResult = calcMaxConcurrentRequests(GLM_47, "fp16", 960, 1000, 500);

    expect(dsResult).toBeGreaterThan(glmResult);
  });

  it("returns 0 for incomplete model", () => {
    expect(calcMaxConcurrentRequests(INCOMPLETE_MODEL, "fp16", 960, 1000, 500)).toBe(0);
  });

  it("returns 0 when model has null learnable_params", () => {
    const m = { ...GLM_47, learnable_params_b: null };
    expect(calcMaxConcurrentRequests(m, "fp16", 960, 1000, 500)).toBe(0);
  });

  it("FP8 KV cache nearly doubles concurrent requests", () => {
    const fp16 = calcMaxConcurrentRequests(GLM_47, "fp16", 960, 1000, 500, 2);
    const fp8 = calcMaxConcurrentRequests(GLM_47, "fp16", 960, 1000, 500, 1);
    // FP8 KV halves per-request memory → roughly doubles concurrency
    expect(fp8).toBeGreaterThan(fp16 * 1.8);
    expect(fp8).toBeLessThan(fp16 * 2.1);
  });

  it("prefix caching (30% hit rate → 0.70 utilization) increases concurrent requests", () => {
    const baseline = calcMaxConcurrentRequests(GLM_47, "fp16", 960, 1000, 500, 2, 1.0);
    const withCaching = calcMaxConcurrentRequests(GLM_47, "fp16", 960, 1000, 500, 2, 0.70);
    // 30% prefix cache hit rate → ~43% more concurrent requests
    expect(withCaching).toBeGreaterThan(baseline);
    expect(withCaching).toBeCloseTo(Math.floor(148.56 / (0.5264 * 0.70)), 0);
  });

  it("FP8 KV + prefix caching stack multiplicatively", () => {
    const baseline = calcMaxConcurrentRequests(GLM_47, "fp16", 960, 1000, 500, 2, 1.0);
    const both = calcMaxConcurrentRequests(GLM_47, "fp16", 960, 1000, 500, 1, 0.70);
    // FP8 (2x) × prefix caching (1/0.7 ≈ 1.43x) ≈ 2.86x
    expect(both).toBeGreaterThan(baseline * 2.5);
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
  it("returns true for NVLink variants", () => {
    expect(isNvLink("NVLink sxm4")).toBe(true);
    expect(isNvLink("NVLink sxm5")).toBe(true);
    expect(isNvLink("nvlink")).toBe(true);
    expect(isNvLink("NVLINK")).toBe(true);
  });

  it("returns false for non-NVLink", () => {
    expect(isNvLink("PCIe")).toBe(false);
    expect(isNvLink(null)).toBe(false);
    expect(isNvLink("")).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// calcParallelismTopology
// ---------------------------------------------------------------------------

describe("calcParallelismTopology", () => {
  it("single-node: all GPUs do TP", () => {
    expect(calcParallelismTopology(1)).toEqual({ tp: 1, pp: 1 });
    expect(calcParallelismTopology(4)).toEqual({ tp: 4, pp: 1 });
    expect(calcParallelismTopology(8)).toEqual({ tp: 8, pp: 1 });
  });

  it("multi-node: TP=8, PP=ceil(gpuCount/8)", () => {
    expect(calcParallelismTopology(10)).toEqual({ tp: 8, pp: 2 });
    expect(calcParallelismTopology(16)).toEqual({ tp: 8, pp: 2 });
    expect(calcParallelismTopology(24)).toEqual({ tp: 8, pp: 3 });
  });
});

// ---------------------------------------------------------------------------
// calcTpEfficiency
// ---------------------------------------------------------------------------

describe("calcTpEfficiency", () => {
  it("returns 1.0 for tp=1", () => {
    expect(calcTpEfficiency(1, null)).toBe(1.0);
    expect(calcTpEfficiency(1, "NVLink sxm5")).toBe(1.0);
  });

  it("NVLink: 5% penalty per doubling", () => {
    expect(calcTpEfficiency(2, "NVLink sxm5")).toBeCloseTo(0.95, 4);
    expect(calcTpEfficiency(4, "NVLink sxm5")).toBeCloseTo(0.90, 4);
    expect(calcTpEfficiency(8, "NVLink sxm5")).toBeCloseTo(0.85, 4);
  });

  it("PCIe: 12% penalty per doubling", () => {
    expect(calcTpEfficiency(2, "PCIe")).toBeCloseTo(0.88, 4);
    expect(calcTpEfficiency(4, "PCIe")).toBeCloseTo(0.76, 4);
    expect(calcTpEfficiency(8, "PCIe")).toBeCloseTo(0.64, 4);
  });

  it("null interconnect treated as PCIe", () => {
    expect(calcTpEfficiency(4, null)).toBeCloseTo(0.76, 4);
  });
});

// ---------------------------------------------------------------------------
// calcPpBubbleEfficiency
// ---------------------------------------------------------------------------

describe("calcPpBubbleEfficiency", () => {
  it("returns 1.0 for pp=1", () => {
    expect(calcPpBubbleEfficiency(1, 10)).toBe(1.0);
  });

  it("batch / (batch + pp - 1)", () => {
    expect(calcPpBubbleEfficiency(2, 10)).toBeCloseTo(10 / 11, 4);
    expect(calcPpBubbleEfficiency(3, 10)).toBeCloseTo(10 / 12, 4);
  });

  it("returns 0 for zero batch size", () => {
    expect(calcPpBubbleEfficiency(2, 0)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// calcDecodeThroughput
// ---------------------------------------------------------------------------

describe("calcDecodeThroughput", () => {
  it("uses active model memory (not total) for MoE throughput", () => {
    // GLM-4.7: MoE, active_params_b=33.7 at fp16 → active_model = 67.4 GB
    // H100 bandwidth = 3.35 TB/s
    // 1 GPU: throughput = 3.35 × 1024 / 67.4 ≈ 50.9 tok/s
    // (Old formula used total 705.6 GB → 4.86 tok/s, ~10x too low)
    const result = calcDecodeThroughput(GLM_47, "fp16", "H100", 1, null);
    expect(result).not.toBeNull();
    expect(result!).toBeCloseTo(3.35 * 1024 / (33.7 * 2), 1);
  });

  it("scales with TP efficiency, not linearly with GPU count", () => {
    const t1 = calcDecodeThroughput(GLM_47, "fp16", "H100", 1, null)!;
    // 4× H100 NVLink: tp=4, tpEff=0.90 → throughput = 4 × 0.90 × t1 = 3.6 × t1
    const t4nvlink = calcDecodeThroughput(GLM_47, "fp16", "H100", 4, "NVLink sxm5")!;
    expect(t4nvlink).toBeCloseTo(t1 * 4 * 0.90, 1);
    // 4× PCIe: tp=4, tpEff=0.76 → throughput = 4 × 0.76 × t1 = 3.04 × t1
    const t4pcie = calcDecodeThroughput(GLM_47, "fp16", "H100", 4, "PCIe")!;
    expect(t4pcie).toBeCloseTo(t1 * 4 * 0.76, 1);
  });

  it("multi-node: 10 GPUs caps TP at 8", () => {
    const t8 = calcDecodeThroughput(GLM_47, "fp16", "H100", 8, "PCIe")!;
    const t10 = calcDecodeThroughput(GLM_47, "fp16", "H100", 10, "PCIe")!;
    // 10 GPUs → tp=8, same throughput as 8 GPUs (PP doesn't add per-stream bandwidth)
    expect(t10).toBeCloseTo(t8, 1);
  });

  it("returns null for unknown GPU", () => {
    expect(calcDecodeThroughput(GLM_47, "fp16", "FakeGPU", 1, null)).toBeNull();
  });

  it("returns null when model memory is null", () => {
    const m = { ...GLM_47, learnable_params_b: null };
    expect(calcDecodeThroughput(m, "fp16", "H100", 1, null)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// calcTeamCapacity — end-to-end integration
// ---------------------------------------------------------------------------

describe("calcTeamCapacity", () => {
  const offering: GpuOffering = {
    gpu_name: "H100",
    vram_gb: 80,
    gpu_count: 12,
    total_vram_gb: 960,
    price_per_hour: 30,
    currency: "USD",
    provider: "test",
    instance_name: "test-instance",
    location: "test-region",
    interconnect: null,
  };

  it("returns positive team size for GLM-4.7 on 12×H100", () => {
    const result = calcTeamCapacity(GLM_47, "fp16", offering, "low-concurrency");
    expect(result.maxConcurrentRequests).toBeGreaterThan(0);
    expect(result.decodeThroughput).toBeGreaterThan(0);
    expect(result.comfortableTeamSize).toBeGreaterThan(0);
    expect(result.costPerUserPerMonth).toBeGreaterThan(0);
    expect(result.costPerUserPerMonth).toBeLessThan(Infinity);
  });

  it("returns zero team for model that doesn't fit", () => {
    // GLM-4.7 FP16 needs 811 GB, 4×H100 = 320 GB — doesn't fit
    const small = { ...offering, gpu_count: 4, total_vram_gb: 320 };
    const result = calcTeamCapacity(GLM_47, "fp16", small, "low-concurrency");
    expect(result.maxConcurrentRequests).toBe(0);
    expect(result.comfortableTeamSize).toBe(0);
  });

  it("hard limit > comfortable team size", () => {
    const result = calcTeamCapacity(GLM_47, "fp16", offering, "low-concurrency");
    if (result.comfortableTeamSize > 0) {
      expect(result.hardLimitTeamSize).toBeGreaterThanOrEqual(result.comfortableTeamSize);
    }
  });

  it("long-context regime uses model.context_length × 0.5 as input tokens", () => {
    // Long-context should produce fewer concurrent requests due to huge KV
    const lowConc = calcTeamCapacity(GLM_47, "fp16", offering, "low-concurrency");
    const longCtx = calcTeamCapacity(GLM_47, "fp16", offering, "long-context");
    expect(longCtx.maxConcurrentRequests).toBeLessThan(lowConc.maxConcurrentRequests);
  });

  it("returns default result for null learnable_params", () => {
    const m = { ...GLM_47, learnable_params_b: null };
    const result = calcTeamCapacity(m, "fp16", offering, "low-concurrency");
    expect(result.maxConcurrentRequests).toBe(0);
    expect(result.comfortableTeamSize).toBe(0);
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
