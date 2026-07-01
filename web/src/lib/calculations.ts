import type {
  Model,
  GpuOffering,
  Precision,
  KvCachePrecision,
  CostResult,
  BenchmarkScore,
} from "@/types";
import { getOfferings } from "./data";
import { supportsFp8KvCache } from "./gpu-specs";

const HOURS_PER_MONTH = 720;

/** Multiplier to account for activation memory, CUDA context, etc. */
export const WEIGHT_OVERHEAD_FACTOR = 1.15;

/**
 * Resolve KV cache precision to bytes per element.
 *
 * - "auto": FP8 (1 byte) if the GPU supports FP8 KV cache, else FP16 (2 bytes)
 * - "fp8": always 1 byte
 * - "fp16": always 2 bytes
 */
export function resolveKvPrecisionBytes(
  kvCachePrecision: KvCachePrecision,
  gpuName: string,
): 1 | 2 {
  if (kvCachePrecision === "fp8") return 1;
  if (kvCachePrecision === "fp16") return 2;
  // "auto"
  return supportsFp8KvCache(gpuName) ? 1 : 2;
}

/** Bytes per parameter for each precision level. */
const BYTES_PER_PARAM: Record<Precision, number> = {
  fp32: 4,
  fp16: 2,
  bf16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5625,
};

/**
 * Resolve the serving precision for a model from its metadata.
 *
 * Maps the model's published precision string (e.g. "FP8", "BF16", "INT4-mixed")
 * to our Precision type. Falls back to "fp16" for unknown values.
 */
export function resolveModelPrecision(model: Model): Precision {
  const raw = (model.precision ?? "").toUpperCase().trim();
  // MXFP8 is an 8-bit format (8-bit values + a shared E8M0 scale per 32-value
  // block, ~1.03 effective bytes/param); size it in the 1-byte fp8 bucket.
  if (raw === "FP8" || raw === "FLOAT8" || raw === "MXFP8") return "fp8";
  if (raw === "BF16") return "bf16";
  if (raw === "FP16") return "fp16";
  if (raw === "FP32") return "fp32";
  if (raw === "INT8") return "int8";
  // "FP4" is DeepSeek-V4's mixed FP4-expert / FP8-rest checkpoint; map the
  // quantized bulk to the int4 byte bucket (see nonRoutedBytesPerParam for
  // the non-expert split).
  if (raw === "FP4" || raw.startsWith("INT4") || raw === "FLOAT4") return "int4";
  return "fp16"; // conservative fallback
}

/**
 * Bytes/param for the *non-routed* weights (attention, shared experts,
 * embeddings, lm_head) of a mixed-precision MoE checkpoint.
 *
 * Community INT4 / NVFP4 quants leave these in BF16; FP4-native checkpoints
 * (e.g. DeepSeek-V4) keep them in FP8. Routed experts always use the int4
 * byte bucket.
 */
function nonRoutedBytesPerParam(model: Model): number {
  const raw = (model.precision ?? "").toUpperCase().trim();
  return raw === "FP4" ? BYTES_PER_PARAM["fp8"] : BYTES_PER_PARAM["bf16"];
}

/** Get memory in GB for a model at a given precision.
 *
 * For mixed-precision models (with routed_expert_params_b), when precision
 * resolves to int4: routed experts use the int4 byte bucket, remaining
 * params (attention, shared experts, lm_head) use BF16 — or FP8 for
 * FP4-native checkpoints like DeepSeek-V4 (see nonRoutedBytesPerParam).
 *
 * This is the TOTAL model size — use for VRAM capacity (does the model fit?).
 * For decode throughput, use getActiveModelMemory instead.
 */
export function getModelMemory(model: Model, precision: Precision): number | null {
  const params = model.learnable_params_b;
  if (params === null) return null;

  if (model.routed_expert_params_b !== null && precision === "int4") {
    const quantizedMem = model.routed_expert_params_b * BYTES_PER_PARAM["int4"];
    const nonQuantizedMem =
      (params - model.routed_expert_params_b) * nonRoutedBytesPerParam(model);
    return quantizedMem + nonQuantizedMem;
  }

  return params * BYTES_PER_PARAM[precision];
}

/**
 * Get active model memory in GB — the bytes read from HBM per decode step.
 *
 * For MoE models, each decode step only reads the active expert weights,
 * not all experts. This is the key advantage of MoE for inference speed:
 * total params determine VRAM capacity, active params determine bandwidth.
 *
 * Falls back to getModelMemory when active_params_b is unavailable or
 * the model is dense (all params are active).
 */
export function getActiveModelMemory(model: Model, precision: Precision): number | null {
  if (model.active_params_b === null || model.architecture !== "MoE") {
    return getModelMemory(model, precision);
  }

  const params = model.learnable_params_b;
  if (params === null) return null;

  // MoE with mixed precision: active routed experts at int4, rest at BF16
  // (or FP8 for FP4-native checkpoints like DeepSeek-V4).
  if (model.routed_expert_params_b !== null && precision === "int4") {
    const nonRoutedParams = params - model.routed_expert_params_b;
    const activeRoutedParams = model.active_params_b - nonRoutedParams;
    return (
      activeRoutedParams * BYTES_PER_PARAM["int4"] +
      nonRoutedParams * nonRoutedBytesPerParam(model)
    );
  }

  // MoE with uniform precision
  return model.active_params_b * BYTES_PER_PARAM[precision];
}

/** Get available precisions for a model. */
export function getAvailablePrecisions(model: Model): Precision[] {
  if (model.learnable_params_b === null) return [];
  return ["fp16", "int8", "int4"];
}

/** Check if a precision is quantized (int8 or int4). */
export function isQuantizedPrecision(precision: Precision): boolean {
  return precision === "int8" || precision === "int4";
}

/** Calculate number of GPUs needed to fit a model's memory requirement. */
export function gpusNeeded(memoryGb: number, vramPerGpu: number): number {
  return Math.ceil(memoryGb / vramPerGpu);
}

/**
 * Find the cheapest per-GPU hourly rate for a given GPU type and region.
 * Returns null if no offerings exist for that type/region.
 */
export function findCheapestPerGpuRate(
  allGpus: GpuOffering[],
  gpuType: string,
): { gpuName: string; perGpuPrice: number } | null {
  const offerings = getOfferings(allGpus, gpuType);
  if (offerings.length === 0) return null;

  let best = offerings[0];
  let bestRate = best.price_per_hour / best.gpu_count;

  for (const o of offerings) {
    const rate = o.price_per_hour / o.gpu_count;
    if (rate < bestRate) {
      best = o;
      bestRate = rate;
    }
  }

  return { gpuName: best.gpu_name, perGpuPrice: bestRate };
}

/** Calculate monthly cost from hourly price. */
export function monthlyCost(pricePerHour: number): number {
  return pricePerHour * HOURS_PER_MONTH;
}

/**
 * Calculate cost results for all models with benchmarks.
 *
 * Returns an array of CostResult sorted by benchmark rank (best first).
 */
export function calculateAllCosts(
  allGpus: GpuOffering[],
  allModels: Model[],
  allBenchmarks: BenchmarkScore[],
  gpuType: string,
  region: string,
  precision: Precision,
): CostResult[] {
  const results: CostResult[] = [];

  for (const benchmark of allBenchmarks) {
    const model = allModels.find((m) => m.model_name === benchmark.model_name);
    if (!model) continue;

    const memGb = getModelMemory(model, precision);
    if (memGb === null) {
      results.push({
        model,
        benchmark,
        memoryGb: null,
        gpusNeeded: null,
        gpuName: null,
        pricePerGpuHour: null,
        monthlyCost: null,
      });
      continue;
    }

    const ref = findCheapestPerGpuRate(allGpus, gpuType);
    if (!ref) {
      results.push({
        model,
        benchmark,
        memoryGb: memGb,
        gpusNeeded: null,
        gpuName: null,
        pricePerGpuHour: null,
        monthlyCost: null,
      });
      continue;
    }

    // Find VRAM per GPU for this type (account for activation/CUDA overhead)
    const sampleOffering = getOfferings(allGpus, gpuType)[0];
    const needed = gpusNeeded(memGb * WEIGHT_OVERHEAD_FACTOR, sampleOffering.vram_gb);

    results.push({
      model,
      benchmark,
      memoryGb: memGb,
      gpusNeeded: needed,
      gpuName: ref.gpuName,
      pricePerGpuHour: ref.perGpuPrice,
      monthlyCost: monthlyCost(needed * ref.perGpuPrice),
    });
  }

  // Sort by benchmark rank (ascending), nulls last
  results.sort((a, b) => {
    const rankA = a.benchmark?.rank ?? Infinity;
    const rankB = b.benchmark?.rank ?? Infinity;
    return rankA - rankB;
  });

  return results;
}

// ============================================================================
// Parallelism Helpers
// ============================================================================

/**
 * Check if an interconnect string refers to an NVLink-based fabric.
 *
 * Real data uses values like "NVLink sxm4", "NVLink sxm5", etc.; the tier
 * enums `nvlink_paired` and `nvswitch` are both NVLink-based too.
 */
export function isNvLink(interconnect: string | null): boolean {
  if (!interconnect) return false;
  const s = interconnect.toLowerCase();
  return s.startsWith("nvlink") || s.startsWith("nvswitch");
}

// ============================================================================
// KV cache
// ============================================================================

/**
 * Calculate KV cache memory per token in GB.
 *
 * Uses architecture-aware formulas:
 * - MLA:  layers × (kv_lora_rank + qk_rope_head_dim) × kvPrecisionBytes
 * - GQA:  2 × layers × num_kv_heads × head_dim × kvPrecisionBytes
 * - DSV4: kv_elems_per_token × kvPrecisionBytes (per-token element width
 *         precomputed in the pipeline; DeepSeek-V4's compressed + sparse-
 *         indexed KV is per-layer-variable and not expressible as MLA)
 * - MSA:  kv_elems_per_token × kvPrecisionBytes (MiniMax-M3 Sparse Attention
 *         keeps a full GQA KV cache plus a per-token block-selection indexer
 *         key cache — MSA saves compute, not KV memory; precomputed in the
 *         pipeline)
 *
 * @param kvPrecisionBytes — 1 for FP8, 2 for FP16 (default: 2 for backward compat)
 * Returns 0 if required fields are missing.
 */
export function calcKvCachePerToken(model: Model, kvPrecisionBytes: 1 | 2 = 2): number {
  const totalLayers = model.num_hidden_layers;
  if (totalLayers === null) return 0;

  // Hybrid models: only some layers have KV cache (e.g. DeltaNet/Mamba2 + Attention)
  const kvLayers = model.num_kv_layers ?? totalLayers;

  let bytesPerToken: number;

  if (model.attention_type === "MLA") {
    if (model.kv_lora_rank === null || model.qk_rope_head_dim === null) return 0;
    bytesPerToken = kvLayers * (model.kv_lora_rank + model.qk_rope_head_dim) * kvPrecisionBytes;
  } else if (model.attention_type === "GQA") {
    if (model.num_kv_heads === null || model.head_dim === null) return 0;
    bytesPerToken = 2 * kvLayers * model.num_kv_heads * model.head_dim * kvPrecisionBytes;
  } else if (model.attention_type === "DSV4" || model.attention_type === "MSA") {
    // Precomputed per-token KV element width from the pipeline. DSV4 =
    // compressed + sparse-indexed KV; MSA (MiniMax-M3) = full GQA KV + block-
    // selection indexer key cache. Both are per-layer-variable and not
    // expressible as plain GQA.
    if (model.kv_elems_per_token === null) return 0;
    bytesPerToken = model.kv_elems_per_token * kvPrecisionBytes;
  } else {
    return 0;
  }

  // Convert bytes to GB
  return bytesPerToken / (1024 * 1024 * 1024);
}
