import type {
  Model,
  GpuOffering,
  Precision,
  KvCachePrecision,
  CostResult,
  BenchmarkScore,
  UsageRegime,
  TeamCapacityResult,
} from "@/types";
import { getOfferings } from "./data";
import { getGpuThroughputSpec, supportsFp8KvCache } from "./gpu-specs";
import { getRegimeParams } from "./regime-params";

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
  if (raw === "FP8") return "fp8";
  if (raw === "BF16") return "bf16";
  if (raw === "FP16") return "fp16";
  if (raw === "FP32") return "fp32";
  if (raw === "INT8") return "int8";
  if (raw.startsWith("INT4")) return "int4";
  return "fp16"; // conservative fallback
}

/** Get memory in GB for a model at a given precision.
 *
 * For mixed-precision models (with routed_expert_params_b), when precision
 * resolves to int4: routed experts use INT4 (0.5 bytes/param), remaining
 * params (attention, shared experts, lm_head) use BF16 (2 bytes/param).
 *
 * This is the TOTAL model size — use for VRAM capacity (does the model fit?).
 * For decode throughput, use getActiveModelMemory instead.
 */
export function getModelMemory(model: Model, precision: Precision): number | null {
  const params = model.learnable_params_b;
  if (params === null) return null;

  if (model.routed_expert_params_b !== null && precision === "int4") {
    const quantizedMem = model.routed_expert_params_b * BYTES_PER_PARAM["int4"];
    const nonQuantizedMem = (params - model.routed_expert_params_b) * BYTES_PER_PARAM["bf16"];
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

  // MoE with INT4 mixed precision: routed experts at INT4, rest at BF16
  if (model.routed_expert_params_b !== null && precision === "int4") {
    const nonRoutedParams = params - model.routed_expert_params_b;
    const activeRoutedParams = model.active_params_b - nonRoutedParams;
    return activeRoutedParams * BYTES_PER_PARAM["int4"] + nonRoutedParams * BYTES_PER_PARAM["bf16"];
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
 * Check if an interconnect string refers to NVLink.
 *
 * Real data uses values like "NVLink sxm4", "NVLink sxm5", etc.
 */
export function isNvLink(interconnect: string | null): boolean {
  if (!interconnect) return false;
  return interconnect.toLowerCase().startsWith("nvlink");
}

/**
 * Determine tensor-parallel (TP) and pipeline-parallel (PP) split.
 *
 * - gpuCount <= 8: all GPUs do TP (single node)
 * - gpuCount > 8: TP=8 (one node), PP = ceil(gpuCount / 8)
 */
export function calcParallelismTopology(gpuCount: number): { tp: number; pp: number } {
  if (gpuCount <= 8) return { tp: gpuCount, pp: 1 };
  return { tp: 8, pp: Math.ceil(gpuCount / 8) };
}

/**
 * TP communication efficiency factor (0–1).
 *
 * Each doubling of TP degree adds one all-reduce, costing:
 * - NVLink: ~5% per doubling  → {2:0.95, 4:0.90, 8:0.85}
 * - PCIe:  ~12% per doubling  → {2:0.88, 4:0.76, 8:0.64}
 *
 * Returns 1.0 for tp=1 (no communication needed).
 */
export function calcTpEfficiency(tp: number, interconnect: string | null): number {
  if (tp <= 1) return 1.0;
  const penaltyPerDoubling = isNvLink(interconnect) ? 0.05 : 0.12;
  return 1.0 - penaltyPerDoubling * Math.log2(tp);
}

/**
 * Pipeline-parallelism bubble efficiency (0–1).
 *
 * PP introduces a bubble: the first and last stages idle while the pipeline
 * fills and drains. With `batchSize` micro-batches and `pp` stages:
 *
 *   efficiency = batchSize / (batchSize + pp - 1)
 *
 * Returns 1.0 when pp=1 (no pipeline parallelism).
 */
export function calcPpBubbleEfficiency(pp: number, batchSize: number): number {
  if (pp <= 1) return 1.0;
  if (batchSize <= 0) return 0;
  return batchSize / (batchSize + pp - 1);
}

// ============================================================================
// Team Capacity & Throughput Calculations
// ============================================================================

/**
 * Calculate KV cache memory per token in GB.
 *
 * Uses architecture-aware formulas:
 * - MLA: layers × (kv_lora_rank + qk_rope_head_dim) × kvPrecisionBytes
 * - GQA: 2 × layers × num_kv_heads × head_dim × kvPrecisionBytes
 *
 * @param kvPrecisionBytes — 1 for FP8, 2 for FP16 (default: 2 for backward compat)
 * Returns 0 if required fields are missing.
 */
export function calcKvCachePerToken(model: Model, kvPrecisionBytes: 1 | 2 = 2): number {
  const layers = model.num_hidden_layers;
  if (layers === null) return 0;

  let bytesPerToken: number;

  if (model.attention_type === "MLA") {
    if (model.kv_lora_rank === null || model.qk_rope_head_dim === null) return 0;
    bytesPerToken = layers * (model.kv_lora_rank + model.qk_rope_head_dim) * kvPrecisionBytes;
  } else if (model.attention_type === "GQA") {
    if (model.num_kv_heads === null || model.head_dim === null) return 0;
    bytesPerToken = 2 * layers * model.num_kv_heads * model.head_dim * kvPrecisionBytes;
  } else {
    return 0;
  }

  // Convert bytes to GB
  return bytesPerToken / (1024 * 1024 * 1024);
}

/**
 * Calculate KV cache memory for a single request in GB.
 *
 * KV cache = (input_tokens + output_tokens) * kv_cache_per_token
 *
 * @param kvPrecisionBytes — 1 for FP8, 2 for FP16 (default: 2)
 */
export function calcKvCachePerRequest(
  model: Model,
  inputTokens: number,
  outputTokens: number,
  kvPrecisionBytes: 1 | 2 = 2,
): number {
  const kvCachePerToken = calcKvCachePerToken(model, kvPrecisionBytes);
  const totalTokens = inputTokens + outputTokens;
  return totalTokens * kvCachePerToken;
}

/**
 * Calculate decode throughput in tokens/sec.
 *
 * Decode is memory-bandwidth limited: throughput ≈ bandwidth / active_model_bytes
 *
 * For MoE models, only active expert weights are read per decode step,
 * so throughput scales with active params, not total params.
 *
 * For multi-GPU setups:
 * - TP GPUs (max 8 per node) contribute memory bandwidth, with communication overhead
 * - PP GPUs hold different pipeline stages and don't add per-stream bandwidth
 *
 * Returns null if GPU throughput specs are unavailable.
 */
export function calcDecodeThroughput(
  model: Model,
  precision: Precision,
  gpuType: string,
  gpuCount: number,
  interconnect: string | null,
): number | null {
  const gpuSpec = getGpuThroughputSpec(gpuType);
  if (!gpuSpec) return null;

  // Use active model memory (MoE reads only active experts per step)
  const activeMemoryGb = getActiveModelMemory(model, precision);
  if (activeMemoryGb === null) return null;

  // Active model size in bytes
  const modelSizeBytes = activeMemoryGb * 1024 * 1024 * 1024;

  // Only TP GPUs contribute bandwidth (PP stages hold different layers)
  const { tp } = calcParallelismTopology(gpuCount);
  const tpEff = calcTpEfficiency(tp, interconnect);

  // Effective bandwidth: TP GPUs × per-GPU bandwidth × TP efficiency
  const totalBandwidthTbS = gpuSpec.memory_bandwidth_tb_s * tp * tpEff;

  // Convert TB/s to bytes/s
  const totalBandwidthBytesPerS = totalBandwidthTbS * 1024 * 1024 * 1024 * 1024;

  // Throughput = bandwidth / model_size
  const throughputTokensPerS = totalBandwidthBytesPerS / modelSizeBytes;

  return throughputTokensPerS;
}

/**
 * Calculate maximum concurrent requests that fit in VRAM.
 *
 * Uses leftover VRAM after weights (with overhead) for KV cache budget:
 *   kvBudget = totalVram - weightMem × WEIGHT_OVERHEAD_FACTOR
 *   effectiveKvPerRequest = kvCachePerRequest × avgCacheUtilization
 *   maxConcurrent = floor(kvBudget / effectiveKvPerRequest)
 *
 * @param kvPrecisionBytes — 1 for FP8, 2 for FP16 (default: 2)
 * @param avgCacheUtilization — fraction of KV actually used per request after
 *   prefix caching (0.1–1.0, default: 1.0 = no caching)
 */
export function calcMaxConcurrentRequests(
  model: Model,
  precision: Precision,
  totalVramGb: number,
  inputTokens: number,
  outputTokens: number,
  kvPrecisionBytes: 1 | 2 = 2,
  avgCacheUtilization: number = 1.0,
): number {
  const modelMemoryGb = getModelMemory(model, precision);
  if (modelMemoryGb === null) return 0;

  const kvCachePerRequest = calcKvCachePerRequest(model, inputTokens, outputTokens, kvPrecisionBytes);
  if (kvCachePerRequest === 0) return 0;

  const effectiveKvPerRequest = kvCachePerRequest * avgCacheUtilization;
  if (effectiveKvPerRequest === 0) return 0;

  const kvBudgetGb = totalVramGb - modelMemoryGb * WEIGHT_OVERHEAD_FACTOR;
  if (kvBudgetGb <= 0) return 0;

  return Math.max(0, Math.floor(kvBudgetGb / effectiveKvPerRequest));
}

/**
 * Calculate team capacity for a given GPU configuration and usage regime.
 *
 * This is the main function that ties everything together.
 *
 * Returns team sizing, cost per user, and utilization metrics.
 */
export function calcTeamCapacity(
  model: Model,
  precision: Precision,
  gpuOffering: GpuOffering,
  regime: UsageRegime,
): TeamCapacityResult {
  const regimeParams = getRegimeParams(regime);
  const modelMemoryGb = getModelMemory(model, precision);

  // Initialize default result
  const defaultResult: TeamCapacityResult = {
    maxConcurrentRequests: 0,
    decodeThroughput: null,
    requestsPerSecond: 0,
    safeRequestsPerSecond: 0,
    comfortableTeamSize: 0,
    hardLimitTeamSize: 0,
    costPerUserPerMonth: Infinity,
    throughputUtilization: 0,
  };

  if (modelMemoryGb === null) return defaultResult;

  // Determine input/output tokens based on regime
  let inputTokens = regimeParams.avgInputTokens;
  if (regime === "long-context" && model.context_length) {
    inputTokens = Math.floor(model.context_length * 0.5);
  }
  const outputTokens = regimeParams.avgOutputTokens;

  // Calculate max concurrent requests
  const maxConcurrentRequests = calcMaxConcurrentRequests(
    model,
    precision,
    gpuOffering.total_vram_gb,
    inputTokens,
    outputTokens,
  );

  if (maxConcurrentRequests === 0) return defaultResult;

  // Calculate decode throughput
  const decodeThroughput = calcDecodeThroughput(
    model,
    precision,
    gpuOffering.gpu_name,
    gpuOffering.gpu_count,
    gpuOffering.interconnect,
  );

  // If we don't have throughput data, we can't calculate team capacity
  if (decodeThroughput === null) {
    return {
      ...defaultResult,
      maxConcurrentRequests,
      decodeThroughput: null,
    };
  }

  const tokensPerRequest = inputTokens + outputTokens;

  // Apply PP bubble penalty when pipeline parallelism is used
  const { pp } = calcParallelismTopology(gpuOffering.gpu_count);
  const ppEff = calcPpBubbleEfficiency(pp, maxConcurrentRequests);

  // System capacity in requests/sec
  // Formula: (decode_throughput × max_concurrent × pp_efficiency) / tokens_per_request
  const requestsPerSecond = (decodeThroughput * maxConcurrentRequests * ppEff) / tokensPerRequest;

  // Safe requests/sec at utilization target
  const safeRequestsPerSecond = requestsPerSecond * regimeParams.utilizationTarget;

  // Team sizing
  // Formula: (safe_requests_per_sec × 3600) / (prompts_per_user_per_hour × burst_factor)
  const requestsPerHour = safeRequestsPerSecond * 3600;
  const userRequestsPerHour = regimeParams.promptsPerUserPerHour * regimeParams.burstFactor;
  const comfortableTeamSize = Math.floor(requestsPerHour / userRequestsPerHour);

  // Hard limit (100% utilization, no burst)
  const hardLimitRequestsPerHour = requestsPerSecond * 3600;
  const hardLimitTeamSize = Math.floor(hardLimitRequestsPerHour / regimeParams.promptsPerUserPerHour);

  // Cost per user
  const gpuMonthlyCost = gpuOffering.price_per_hour * HOURS_PER_MONTH;
  const costPerUserPerMonth = comfortableTeamSize > 0 ? gpuMonthlyCost / comfortableTeamSize : Infinity;

  // Throughput utilization at comfortable capacity
  const actualRequestsPerSecond = (comfortableTeamSize * userRequestsPerHour) / 3600;
  const throughputUtilization = requestsPerSecond > 0 ? (actualRequestsPerSecond / requestsPerSecond) * 100 : 0;

  return {
    maxConcurrentRequests,
    decodeThroughput,
    requestsPerSecond,
    safeRequestsPerSecond,
    comfortableTeamSize,
    hardLimitTeamSize,
    costPerUserPerMonth,
    throughputUtilization,
  };
}
