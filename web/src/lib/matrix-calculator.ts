import type {
  Model,
  GpuOffering,
  BenchmarkScore,
  SotaScore,
  MatrixCell,
  GpuSetupOption,
  AdvancedSettings,
  PresetGpuConfig,
  ConcurrencyTierConfig,
  DeploymentEstimate,
} from "@/types";
import { CONCURRENCY_TIERS } from "./concurrency-tiers";
import { PERFORMANCE_COLUMNS } from "./performance-columns";
import {
  getModelMemory,
  getActiveModelMemory,
  gpusNeeded,
  resolveModelPrecision,
  resolveKvPrecisionBytes,
  WEIGHT_OVERHEAD_FACTOR,
} from "./calculations";
import { calcTopology } from "./calc-topology";
import {
  calcDecodeLatency,
  type DecodeLatencyGpu,
  type DecodeLatencyModelDims,
} from "./calc-decode-latency";
import { calcRuntimeReserve, calcOperatingStreams } from "./calc-capacity";
import { throughputState } from "./throughput-support";
import { getGpuThroughputSpec } from "./gpu-specs";
import { computeTotalBenchmarkCost } from "./benchmark-costs";
import { scoreFor, minVramForModel, isUnranked } from "./model-data";


const HOURS_PER_MONTH = 720;

/**
 * Default GPU memory utilization (0–1), matching vLLM's --gpu-memory-utilization.
 *
 * Controls what fraction of total VRAM is available for weights + KV cache.
 * The remaining headroom covers CUDA context, PagedAttention block fragmentation,
 * memory allocator overhead, and runtime state.
 */
export const DEFAULT_MEMORY_UTILIZATION = 0.90;

export const DEFAULT_ADVANCED_SETTINGS: AdvancedSettings = {
  avgInputTokens: 50_000,
  avgOutputTokens: 1000,
  prefixReuse: 0.90,
  minTokPerStream: 20,
  // 2 bytes/elem: vLLM's `kv_cache_dtype=auto` follows the model's bf16/fp16
  // compute dtype, not the GPU's FP8 capability. FP8 KV is opt-in ("auto").
  kvCachePrecision: "fp16",
};

interface GpuSetupStats {
  maxConcurrentStreams: number;
  decodeThroughputTokS: number | null;
  deploymentEstimate: DeploymentEstimate | null;
}

/**
 * A GPU setup's headline numbers, derived entirely from the first-principles
 * {@link calcDeploymentEstimate} (F-4): the dtype-correct operating-streams floor
 * (admission capacity) and the architecture-gated single-stream decode rate
 * (null when throughput isn't modeled). The full estimate is returned too, so
 * callers reuse it instead of recomputing.
 *
 * The memoryUtilization parameter (0–1) mirrors vLLM's --gpu-memory-utilization.
 */
export function calcGpuSetupStats(
  model: Model,
  gpuName: string,
  gpuCount: number,
  totalVramGb: number,
  interconnect: string | null,
  settings: AdvancedSettings,
  memoryUtilization: number = DEFAULT_MEMORY_UTILIZATION,
): GpuSetupStats {
  const vramPerGpu = gpuCount > 0 ? totalVramGb / gpuCount : totalVramGb;
  const deploymentEstimate = estimateForLayout(
    model,
    gpuName,
    gpuCount,
    vramPerGpu,
    totalVramGb,
    interconnect,
    settings,
    memoryUtilization,
  );
  return {
    // Conservative admission floor: the low end of the operating-streams band.
    maxConcurrentStreams: deploymentEstimate?.operatingStreams.low ?? 0,
    decodeThroughputTokS: deploymentEstimate?.singleStreamTokS ?? null,
    deploymentEstimate,
  };
}

/**
 * Forward-pass FLOPs per parameter per token. A matmul costs one multiply plus
 * one add per weight, so a single forward pass is ~2·N FLOPs per token for an
 * N-parameter model. Standard transformer FLOP accounting (Kaplan et al. 2020,
 * "Scaling Laws for Neural Language Models", Appendix; Hoffmann et al. 2022,
 * "Chinchilla", Appendix F). Public, architecture-independent constant.
 */
const FORWARD_FLOPS_PER_PARAM = 2;

/**
 * Pick the effective unidirectional inter-GPU link bandwidth (GB/s) the decode
 * latency model should charge collectives against, given the GPU's normalized
 * interconnect tier. NVLink/NVSwitch tiers use the NVLink figure; PCIe-only
 * ("none") uses the PCIe link bandwidth. Falls back to whichever datasheet
 * figure is present. Returns null when neither is published.
 */
function interconnectBandwidthGbS(
  spec: { nvlink_bandwidth_gb_s: number | null; pcie_bandwidth_gb_s: number | null },
  tier: "none" | "nvlink_paired" | "nvswitch",
): number | null {
  if (tier === "none") return spec.pcie_bandwidth_gb_s ?? spec.nvlink_bandwidth_gb_s;
  return spec.nvlink_bandwidth_gb_s ?? spec.pcie_bandwidth_gb_s;
}

/**
 * Assemble a first-principles {@link DeploymentEstimate} for one (model, GPU
 * offering) pairing.
 *
 * The pipeline composes the three sibling physics models:
 *  1. {@link calcTopology} chooses the parallelism layout (TP/PP) the
 *     interconnect tier allows, and reports whether the weights even fit.
 *  2. {@link calcDecodeLatency} gives `singleStreamTokS` — the latency-bound
 *     decode rate of one in-flight request under that layout.
 *  3. {@link calcRuntimeReserve} + {@link calcOperatingStreams} give the
 *     low/high concurrency band the leftover VRAM can admit.
 *
 * `aggregateTokS` is the bonus throughput when the layout runs its full
 * operating batch: the *batched* bandwidth roofline (the raw single-stream
 * weight-read roofline reused as a per-step ceiling, amortized over the high
 * operating batch) capped by the prefill compute roofline. The raw roofline
 * stays internal — it is not surfaced on the estimate.
 *
 * Returns null when the GPU specs, required model dims, or layout feasibility
 * make a first-principles estimate impossible (no faked fallbacks).
 */
export function calcDeploymentEstimate(
  model: Model,
  offering: GpuOffering,
  settings: AdvancedSettings,
  memoryUtilization: number = DEFAULT_MEMORY_UTILIZATION,
): DeploymentEstimate | null {
  const spec = getGpuThroughputSpec(offering.gpu_name);
  if (!spec) return null;

  const precision = resolveModelPrecision(model);
  const weightsGb = getModelMemory(model, precision);
  if (weightsGb === null) return null;

  const isMoe = model.architecture === "MoE";

  // 1. Parallelism layout from the interconnect topology (needed for streams).
  const tier = spec.interconnect_tier;
  const layout = calcTopology({
    interconnectTier: tier,
    gpuCount: offering.gpu_count,
    modelSizeGb: weightsGb,
    vramPerGpuGb: offering.vram_gb,
  });
  if (!layout.feasible) return null;

  // 2. Operating-streams band — pure VRAM accounting, robust for EVERY
  // architecture (the only architecture input is KV-bytes-per-token). Computed
  // unconditionally; the throughput block below is what's architecture-gated.
  // For MoE the experts are sharded across the TP group, so EP degree = TP.
  const reserveLayout = {
    numGpus: layout.gpusUsed,
    tp: layout.tp,
    ep: isMoe ? layout.tp : 1,
    pp: layout.pp,
  };
  const reserveGb = calcRuntimeReserve(model, reserveLayout);
  const usableVramGb = layout.gpusUsed * offering.vram_gb * memoryUtilization;
  const kvPrecisionBytes = resolveKvPrecisionBytes(settings.kvCachePrecision, offering.gpu_name);
  // Streams are evaluated at exactly the configured prefix reuse. With GPU-memory
  // utilization fixed at 90% and the context window taken from settings, prefix
  // reuse is the only remaining free variable in the KV budget — and it is itself
  // a single setting — so the formula yields one stream count, not a band. A
  // zero-width range collapses calcOperatingStreams' low/high to that single point.
  const prefixReuseRange = { low: settings.prefixReuse, high: settings.prefixReuse };
  const operatingStreams = calcOperatingStreams({
    model,
    usableVramGb,
    weightsGb,
    reserveGb,
    avgInputTokens: settings.avgInputTokens,
    avgOutputTokens: settings.avgOutputTokens,
    kvPrecisionBytes,
    prefixReuseRange,
  });

  // 3. Throughput overlay — only for architectures the decode-latency model can
  // represent (standard MoE/Dense + GQA/MLA). Null for hybrids / sparse attention.
  const tState = throughputState(model);
  let singleStreamTokS: number | null = null;
  let aggregateTokS: number | null = null;

  const activeParamsB = isMoe ? model.active_params_b : model.learnable_params_b;
  const activeMemGb = getActiveModelMemory(model, precision);
  const interconnectGbS = interconnectBandwidthGbS(spec, tier);
  const hasLinkForCollectives = !(layout.tp > 1 && (interconnectGbS === null || interconnectGbS <= 0));

  if (
    tState === "modeled" &&
    model.num_hidden_layers !== null &&
    model.hidden_size !== null &&
    activeParamsB !== null &&
    activeParamsB > 0 &&
    activeMemGb !== null &&
    hasLinkForCollectives
  ) {
    // Effective bytes/param at the serving precision — captures the mixed-precision
    // (int4 routed experts + bf16/fp8 rest) split that getActiveModelMemory encodes.
    const bytesPerParam = activeMemGb / activeParamsB;
    const latencyGpu: DecodeLatencyGpu = {
      hbmBandwidthTbS: spec.memory_bandwidth_tb_s,
      interconnectTier: tier,
      interconnectBandwidthGbS: interconnectGbS ?? 0,
    };
    const latencyDims: DecodeLatencyModelDims = {
      numLayers: model.num_hidden_layers,
      hiddenSize: model.hidden_size,
      activeParamsB,
      bytesPerParam,
      isMoe,
      topK: model.experts_per_token,
      numExperts: model.num_experts,
      routedExpertParamsB: model.routed_expert_params_b,
      kvLoraRank: model.kv_lora_rank,
      qkRopeHeadDim: model.qk_rope_head_dim,
    };
    // EP degree = TP for MoE (experts sharded across the TP group); PP from layout.
    const decode = calcDecodeLatency(latencyGpu, latencyDims, {
      tp: layout.tp,
      ep: isMoe ? layout.tp : 1,
      pp: layout.pp,
    });
    singleStreamTokS = decode.singleStreamTokS;

    // Aggregate (bonus) throughput: the batched bandwidth roofline capped by the
    // prefill compute roofline (FLOPs across the used GPUs).
    const batchedRooflineTokS = operatingStreams.high * decode.bandwidthRooflineTokS;
    const prefillComputeTokS =
      (layout.gpusUsed * spec.fp16_tflops * 1e12) /
      (FORWARD_FLOPS_PER_PARAM * activeParamsB * 1e9);
    aggregateTokS = Math.min(batchedRooflineTokS, prefillComputeTokS);
  }

  return {
    singleStreamTokS,
    operatingStreams,
    aggregateTokS,
    throughputState: tState,
    assumptions: {
      context: {
        avgInputTokens: settings.avgInputTokens,
        avgOutputTokens: settings.avgOutputTokens,
        prefixReuse: settings.prefixReuse,
      },
      interconnectTier: tier,
      moe: isMoe,
    },
  };
}

/**
 * Build the read-only {@link DeploymentEstimate} for a GPU layout described by
 * its raw primitives (the budget/scaled paths don't carry a full
 * {@link GpuOffering}). `calcDeploymentEstimate` only reads gpu_name, gpu_count
 * and vram_gb off the offering, so the remaining fields are placeholders.
 */
function estimateForLayout(
  model: Model,
  gpuName: string,
  gpuCount: number,
  vramPerGpu: number,
  totalVramGb: number,
  interconnect: string | null,
  settings: AdvancedSettings,
  memoryUtilization: number = DEFAULT_MEMORY_UTILIZATION,
): DeploymentEstimate | null {
  const offering: GpuOffering = {
    gpu_name: gpuName,
    vram_gb: vramPerGpu,
    gpu_count: gpuCount,
    total_vram_gb: totalVramGb,
    price_per_hour: 0,
    currency: "",
    provider: "",
    instance_name: "",
    location: "",
    interconnect,
  };
  return calcDeploymentEstimate(model, offering, settings, memoryUtilization);
}

/**
 * Find GPU offerings that can fit a model and serve a given concurrency level.
 * Returns up to `limit` options sorted by monthly cost (cheapest first).
 *
 * Two checks per offering:
 * 1. KV budget: maxConcurrentStreams >= concurrency
 * 2. Throughput quality: per-stream tok/s >= minTokPerStream
 */
export function findGpuSetups(
  model: Model,
  allGpus: GpuOffering[],
  concurrency: number,
  settings: AdvancedSettings,
  limit: number = 3,
): GpuSetupOption[] {
  const precision = resolveModelPrecision(model);
  const modelMemoryGb = getModelMemory(model, precision);
  if (modelMemoryGb === null) return [];

  // Group by gpu_name + gpu_count to find cheapest per config
  const configMap = new Map<string, GpuOffering>();
  for (const g of allGpus) {
    const key = `${g.gpu_name}|${g.gpu_count}`;
    const existing = configMap.get(key);
    if (!existing || g.price_per_hour < existing.price_per_hour) {
      configMap.set(key, g);
    }
  }

  const options: GpuSetupOption[] = [];

  for (const offering of configMap.values()) {
    // Check if the model fits (account for activation/CUDA overhead)
    const needed = gpusNeeded(modelMemoryGb * WEIGHT_OVERHEAD_FACTOR, offering.vram_gb);
    if (needed > offering.gpu_count) continue;

    const interconnect = offering.interconnect;
    const stats = calcGpuSetupStats(
      model,
      offering.gpu_name,
      offering.gpu_count,
      offering.total_vram_gb,
      interconnect,
      settings,
    );

    // Check 1: KV budget — can it handle the concurrency?
    if (stats.maxConcurrentStreams < concurrency) continue;

    // Check 2: Throughput quality — steady-state per-stream tok/s
    if (stats.decodeThroughputTokS !== null) {
      if (stats.decodeThroughputTokS < settings.minTokPerStream) continue;
    }
    const offeringMonthlyCost = offering.price_per_hour * HOURS_PER_MONTH;
    const costPerStream = concurrency > 0 ? offeringMonthlyCost / concurrency : Infinity;

    options.push({
      gpuName: offering.gpu_name,
      gpuCount: offering.gpu_count,
      interconnect,
      totalVramGb: offering.total_vram_gb,
      monthlyCost: offeringMonthlyCost,
      costPerStreamPerMonth: costPerStream,
      decodeThroughputTokS: stats.decodeThroughputTokS,
      maxConcurrentStreams: stats.maxConcurrentStreams,
      deploymentEstimate: stats.deploymentEstimate,
    });
  }

  // Sort by monthly cost, take cheapest
  options.sort((a, b) => a.monthlyCost - b.monthlyCost);
  return options.slice(0, limit);
}

/**
 * Hard cap for projected GPU scaling — we won't extrapolate beyond this.
 */
const MAX_PROJECTED_GPUS = 8;

/**
 * Fallback for the performance persona: scale GPU count beyond real offerings
 * to find a configuration that can serve the requested concurrency level.
 *
 * Only considers GPU types that have a x1 (single-GPU) offering in the data.
 * Uses the x1 offering's price as the per-GPU rate and caps scaling at 8 GPUs.
 * All returned setups are marked with `isProjected: true`.
 */
export function findScaledGpuSetups(
  model: Model,
  allGpus: GpuOffering[],
  concurrency: number,
  settings: AdvancedSettings,
  limit: number = 3,
): GpuSetupOption[] {
  const precision = resolveModelPrecision(model);
  const modelMemoryGb = getModelMemory(model, precision);
  if (modelMemoryGb === null) return [];

  // Only consider GPU types that have a x1 (single-GPU) offering.
  // Use the cheapest x1 offering's price as the per-GPU rate.
  const gpuTypeMap = new Map<string, { perGpuPrice: number; vramGb: number; interconnect: string | null }>();
  for (const g of allGpus) {
    if (g.gpu_count !== 1) continue;
    const existing = gpuTypeMap.get(g.gpu_name);
    if (!existing || g.price_per_hour < existing.perGpuPrice) {
      gpuTypeMap.set(g.gpu_name, {
        perGpuPrice: g.price_per_hour,
        vramGb: g.vram_gb,
        interconnect: g.interconnect,
      });
    }
  }

  const options: GpuSetupOption[] = [];

  for (const [gpuName, info] of gpuTypeMap) {
    const minGpus = gpusNeeded(modelMemoryGb * WEIGHT_OVERHEAD_FACTOR, info.vramGb);

    for (let count = minGpus; count <= MAX_PROJECTED_GPUS; count++) {
      const totalVram = count * info.vramGb;
      const interconnect = info.interconnect;
      const stats = calcGpuSetupStats(
        model, gpuName, count, totalVram, interconnect, settings,
      );

      if (stats.maxConcurrentStreams < concurrency) continue;

      if (stats.decodeThroughputTokS !== null) {
        if (stats.decodeThroughputTokS < settings.minTokPerStream) continue;
      }
      const scaledMonthlyCost = info.perGpuPrice * count * HOURS_PER_MONTH;
      const costPerStream = concurrency > 0 ? scaledMonthlyCost / concurrency : Infinity;
      options.push({
        gpuName,
        gpuCount: count,
        interconnect,
        totalVramGb: totalVram,
        monthlyCost: scaledMonthlyCost,
        costPerStreamPerMonth: costPerStream,
        decodeThroughputTokS: stats.decodeThroughputTokS,
        maxConcurrentStreams: stats.maxConcurrentStreams,
        isProjected: true,
        deploymentEstimate: stats.deploymentEstimate,
      });
      break;
    }
  }

  options.sort((a, b) => a.monthlyCost - b.monthlyCost);
  return options.slice(0, limit);
}

/**
 * Build the Performance-persona cells for one model: one per
 * {@link PERFORMANCE_COLUMNS} entry (Fit, Scale). Each column picks the cheapest
 * GPU setup that admits at least the column's stream floor (real offerings
 * first, then the 8-GPU scaled fallback), all at the fixed 90% memory
 * utilization the calculator assumes.
 *
 * `base` carries the score-dependent fields (benchmark, sotaScore, percentOfSota,
 * totalBenchmarkCost, isUnranked) that are identical across both columns; they
 * are left null for unranked models.
 */
function performanceCellsForModel(
  model: Model,
  allGpus: GpuOffering[],
  settings: AdvancedSettings,
  base: Pick<
    MatrixCell,
    "benchmark" | "sotaScore" | "percentOfSota" | "totalBenchmarkCost" | "isUnranked"
  >,
): MatrixCell[] {
  return PERFORMANCE_COLUMNS.map((col): MatrixCell => {
    let gpuSetups = findGpuSetups(model, allGpus, col.minStreams, settings, 1);
    if (gpuSetups.length === 0) {
      gpuSetups = findScaledGpuSetups(model, allGpus, col.minStreams, settings, 1);
    }
    const cheapest = gpuSetups.length > 0 ? gpuSetups[0] : null;

    return {
      model,
      ...base,
      gpuSetups,
      costPerStreamPerMonth: cheapest?.costPerStreamPerMonth ?? null,
      exceedsCapacity: gpuSetups.length === 0,
      decodeThroughputTokS: cheapest?.decodeThroughputTokS ?? null,
      // Utilization is fixed at 90% (the memory-utilization the streams band is
      // sized under); it is no longer a per-cell number the UI surfaces.
      utilization: null,
    };
  });
}

/**
 * Calculate the recommendation matrix for the Performance persona.
 *
 * Returns a 2D array: rows = models (sorted by score desc), columns = the two
 * operating points (Fit, Scale) from {@link PERFORMANCE_COLUMNS}.
 */
export function calculatePerformanceMatrix(
  allGpus: GpuOffering[],
  allModels: Model[],
  benchmarks: BenchmarkScore[],
  sotaScores: SotaScore[],
  benchmarkCategory: string,
  settings: AdvancedSettings = DEFAULT_ADVANCED_SETTINGS,
): MatrixCell[][] {
  // Filter benchmarks for the selected category
  const categoryBenchmarks = benchmarks.filter(
    (b) => b.benchmark_name === benchmarkCategory,
  );

  // Find matching SOTA score
  const sota = sotaScores.find((s) => s.benchmark_name === benchmarkCategory) ?? null;

  // Match benchmarks to our models and sort by score descending
  const modelScores: { model: Model; benchmark: BenchmarkScore }[] = [];
  for (const b of categoryBenchmarks) {
    if (b.score === null) continue;
    const model = allModels.find((m) => m.model_name === b.model_name);
    if (!model) continue;
    modelScores.push({ model, benchmark: b });
  }
  modelScores.sort((a, b) => (b.benchmark.score ?? 0) - (a.benchmark.score ?? 0));

  // Build matrix
  return modelScores.map(({ model, benchmark }) => {
    const percentOfSota =
      sota && benchmark.score !== null && sota.sota_score > 0
        ? benchmark.score / sota.sota_score
        : 0;

    const totalBenchmarkCost = computeTotalBenchmarkCost(
      model.model_name, benchmarkCategory, benchmarks,
    );

    return performanceCellsForModel(model, allGpus, settings, {
      benchmark,
      sotaScore: sota,
      percentOfSota,
      totalBenchmarkCost,
      isUnranked: false,
    });
  });
}

/**
 * Calculate the recommendation matrix for the Budget persona.
 *
 * User has a fixed GPU config; we show what models fit and their cost per stream.
 */
export function calculateBudgetMatrix(
  gpuConfig: PresetGpuConfig,
  allGpus: GpuOffering[],
  allModels: Model[],
  benchmarks: BenchmarkScore[],
  sotaScores: SotaScore[],
  benchmarkCategory: string,
  settings: AdvancedSettings = DEFAULT_ADVANCED_SETTINGS,
): MatrixCell[][] {
  const categoryBenchmarks = benchmarks.filter(
    (b) => b.benchmark_name === benchmarkCategory,
  );

  const sota = sotaScores.find((s) => s.benchmark_name === benchmarkCategory) ?? null;

  // Match benchmarks to models, filter to those that fit in the GPU config
  const modelScores: { model: Model; benchmark: BenchmarkScore }[] = [];
  for (const b of categoryBenchmarks) {
    if (b.score === null) continue;
    const model = allModels.find((m) => m.model_name === b.model_name);
    if (!model) continue;

    const memGb = getModelMemory(model, resolveModelPrecision(model));
    if (memGb === null) continue;

    // Check if model fits (account for activation/CUDA overhead)
    const needed = gpusNeeded(memGb * WEIGHT_OVERHEAD_FACTOR, gpuConfig.vramPerGpu);
    if (needed > gpuConfig.gpuCount) continue;

    modelScores.push({ model, benchmark: b });
  }
  modelScores.sort((a, b) => (b.benchmark.score ?? 0) - (a.benchmark.score ?? 0));

  // Find the cheapest per-GPU rate for this GPU type in the region,
  // then extrapolate to the configured count. This way configs like
  // 4× H200 work even if the data only has 1× and 2× H200 offerings.
  const gpuOfferings = allGpus.filter((g) => g.gpu_name === gpuConfig.gpuName);
  let bestPerGpuRate = Infinity;
  for (const o of gpuOfferings) {
    const rate = o.price_per_hour / o.gpu_count;
    if (rate < bestPerGpuRate) bestPerGpuRate = rate;
  }
  const monthlyCost = bestPerGpuRate < Infinity
    ? bestPerGpuRate * gpuConfig.gpuCount * HOURS_PER_MONTH
    : null;

  return modelScores.map(({ model, benchmark }) => {
    const percentOfSota =
      sota && benchmark.score !== null && sota.sota_score > 0
        ? benchmark.score / sota.sota_score
        : 0;

    const totalBenchmarkCost = computeTotalBenchmarkCost(
      model.model_name, benchmarkCategory, benchmarks,
    );

    // Compute stats once per model (same GPU config for all tiers)
    const stats = calcGpuSetupStats(
      model,
      gpuConfig.gpuName,
      gpuConfig.gpuCount,
      gpuConfig.totalVramGb,
      gpuConfig.interconnect,
      settings,
    );

    // The deployment estimate is tier-independent (same GPU layout) — reuse the
    // one calcGpuSetupStats already computed.
    const deploymentEstimate = stats.deploymentEstimate;

    return CONCURRENCY_TIERS.map((tier: ConcurrencyTierConfig): MatrixCell => {
      const exceedsCapacity = stats.maxConcurrentStreams < tier.midpoint;

      // Check throughput quality floor — steady-state decode (pipeline always full)
      let throughputTooLow = false;
      if (stats.decodeThroughputTokS !== null) {
        if (stats.decodeThroughputTokS < settings.minTokPerStream) {
          throughputTooLow = true;
        }
      }

      const cannotServe = exceedsCapacity || throughputTooLow;

      let costPerStream: number | null = null;
      if (!cannotServe && monthlyCost !== null && tier.midpoint > 0) {
        costPerStream = monthlyCost / tier.midpoint;
      }

      const gpuSetup: GpuSetupOption | null =
        monthlyCost !== null
          ? {
              gpuName: gpuConfig.gpuName,
              gpuCount: gpuConfig.gpuCount,
              interconnect: gpuConfig.interconnect,
              totalVramGb: gpuConfig.totalVramGb,
              monthlyCost: monthlyCost,
              costPerStreamPerMonth: costPerStream ?? Infinity,
              decodeThroughputTokS: stats.decodeThroughputTokS,
              maxConcurrentStreams: stats.maxConcurrentStreams,
              deploymentEstimate,
            }
          : null;

      return {
        model,
        benchmark,
        sotaScore: sota,
        percentOfSota,
        totalBenchmarkCost,
        gpuSetups: gpuSetup ? [gpuSetup] : [],
        costPerStreamPerMonth: costPerStream,
        exceedsCapacity: cannotServe,
        decodeThroughputTokS: cannotServe ? null : stats.decodeThroughputTokS,
        utilization: stats.maxConcurrentStreams > 0
          ? Math.min(tier.midpoint / stats.maxConcurrentStreams, 1.0)
          : null,
        isUnranked: false,
      };
    });
  });
}

// ============================================================================
// Budget Chart Data (team-size view)
// ============================================================================

export interface BudgetChartDataPoint {
  modelName: string;
  maxConcurrentStreams: number;
  requestsPerHour: number | null;
  // null for unranked models — never coerce a missing score to a number.
  percentOfSota: number | null;
  modelMemoryGb: number;
  fits: boolean;
  doesntFitReason: string | null;
  decodeThroughputTokS: number | null;
  benchmarkScore: number | null;
  // Sized but with no OpenHands Index score yet (partial-model-data skill).
  isUnranked: boolean;
  // First-principles throughput estimate for this model on the fixed GPU config.
  // null when the model doesn't fit or specs/dims make an estimate impossible.
  // Rendered read-only by the budget chart — never recomputed in the view.
  deploymentEstimate: DeploymentEstimate | null;
}

export function calculateBudgetChartData(
  gpuConfig: PresetGpuConfig,
  allModels: Model[],
  benchmarks: BenchmarkScore[],
  sotaScores: SotaScore[],
  benchmarkCategory: string,
  memoryUtilization: number,
  settings: AdvancedSettings,
): BudgetChartDataPoint[] {
  const sota = sotaScores.find((s) => s.benchmark_name === benchmarkCategory) ?? null;

  // Budget is a sizing-first persona: iterate the MODEL list and treat the
  // benchmark score as an optional left join (partial-model-data skill). Ranked
  // models must clear a ≥50%-of-SOTA quality bar; unranked models (sized, no
  // score) are always included. Unsized models can't be placed here at all.
  const entries: { model: Model; benchmark: BenchmarkScore | null }[] = [];
  for (const model of allModels) {
    // Skip models we can't size — Budget excludes unsized models by design.
    if (minVramForModel(model) === null) continue;

    const benchmark = scoreFor(model, benchmarkCategory, benchmarks);
    if (benchmark === null) {
      // Unranked: sized but no score yet — always surfaced.
      entries.push({ model, benchmark: null });
      continue;
    }

    // Ranked: keep only models at ≥50% of SOTA (same bar as before).
    if (!sota || benchmark.score === null || sota.sota_score <= 0) continue;
    if ((benchmark.score / sota.sota_score) * 100 < 50) continue;
    entries.push({ model, benchmark });
  }

  // Sort ranked first by score descending, then unranked grouped after, ordered
  // by minimum VRAM ascending (a sizing proxy). Never let a null score sort as 0.
  entries.sort((a, b) => {
    const aRanked = a.benchmark !== null;
    const bRanked = b.benchmark !== null;
    if (aRanked && bRanked) {
      return (b.benchmark!.score ?? 0) - (a.benchmark!.score ?? 0);
    }
    if (aRanked !== bRanked) return aRanked ? -1 : 1;
    return (minVramForModel(a.model) ?? 0) - (minVramForModel(b.model) ?? 0);
  });

  return entries.map(({ model, benchmark }): BudgetChartDataPoint => {
    const isUnranked = benchmark === null;
    const precision = resolveModelPrecision(model);
    const modelMemoryGb = getModelMemory(model, precision);

    // Check if model physically fits
    const physicallyFits = modelMemoryGb !== null
      ? gpusNeeded(modelMemoryGb * WEIGHT_OVERHEAD_FACTOR, gpuConfig.vramPerGpu) <= gpuConfig.gpuCount
      : false;

    const percentOfSota =
      benchmark !== null && benchmark.score !== null && sota && sota.sota_score > 0
        ? (benchmark.score / sota.sota_score) * 100
        : null;

    if (!physicallyFits || modelMemoryGb === null) {
      return {
        modelName: model.model_name,
        maxConcurrentStreams: 0,
        requestsPerHour: null,
        percentOfSota,
        modelMemoryGb: modelMemoryGb ?? 0,
        fits: false,
        doesntFitReason: "Not enough VRAM",
        decodeThroughputTokS: null,
        benchmarkScore: benchmark?.score ?? null,
        isUnranked,
        deploymentEstimate: null,
      };
    }

    // Compute stats with memory utilization applied to VRAM
    const stats = calcGpuSetupStats(
      model,
      gpuConfig.gpuName,
      gpuConfig.gpuCount,
      gpuConfig.totalVramGb,
      gpuConfig.interconnect,
      settings,
      memoryUtilization,
    );

    const fits = stats.maxConcurrentStreams > 0;

    const doesntFitReason: string | null = fits ? null : "Not enough VRAM for KV cache";

    const requestsPerHour =
      fits && stats.decodeThroughputTokS !== null && stats.decodeThroughputTokS > 0
        ? stats.maxConcurrentStreams * stats.decodeThroughputTokS / settings.avgOutputTokens * 3600
        : null;

    return {
      modelName: model.model_name,
      maxConcurrentStreams: stats.maxConcurrentStreams,
      requestsPerHour,
      percentOfSota,
      modelMemoryGb,
      fits,
      doesntFitReason,
      decodeThroughputTokS: stats.decodeThroughputTokS,
      benchmarkScore: benchmark?.score ?? null,
      isUnranked,
      deploymentEstimate: stats.deploymentEstimate,
    };
  });
}

// ============================================================================
// Unranked Models (Performance persona)
// ============================================================================

/**
 * Build a recommendation matrix for unranked models, reusing the same row/cell
 * shape (and therefore the same `RecommendationMatrix` UI) as the ranked Top
 * Coding Models table.
 *
 * The ranked matrix is ranking-first and iterates benchmark scores, so it can't
 * place unranked models (no score to rank by). This sizing-first helper instead
 * iterates the MODEL list, keeps the ones that are sized but unranked in this
 * category, and computes the cheapest GPU setup per concurrency tier (real
 * offerings first, then the same 8-GPU scaled fallback). Score-dependent cell
 * fields (benchmark, percentOfSota, sotaScore, totalBenchmarkCost) are left
 * null — the UI renders them as gaps; nothing is faked.
 *
 * Rows are sorted by minimum VRAM descending (biggest first).
 */
export function calculateUnrankedMatrix(
  allGpus: GpuOffering[],
  allModels: Model[],
  benchmarks: BenchmarkScore[],
  benchmarkCategory: string,
  settings: AdvancedSettings = DEFAULT_ADVANCED_SETTINGS,
): MatrixCell[][] {
  const unrankedModels = allModels.filter(
    (model) =>
      minVramForModel(model) !== null &&
      isUnranked(model, benchmarkCategory, benchmarks),
  );

  // Biggest VRAM footprint first.
  unrankedModels.sort(
    (a, b) => (minVramForModel(b) ?? 0) - (minVramForModel(a) ?? 0),
  );

  return unrankedModels.map((model) =>
    performanceCellsForModel(model, allGpus, settings, {
      benchmark: null,
      sotaScore: null,
      percentOfSota: null,
      totalBenchmarkCost: null,
      isUnranked: true,
    }),
  );
}
