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
} from "@/types";
import { CONCURRENCY_TIERS } from "./concurrency-tiers";
import {
  getModelMemory,
  gpusNeeded,
  calcDecodeThroughput,
  calcMaxConcurrentRequests,
  resolveModelPrecision,
  resolveKvPrecisionBytes,
  WEIGHT_OVERHEAD_FACTOR,
} from "./calculations";
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
  minTokPerStream: 20,
  // 2 bytes/elem: vLLM's `kv_cache_dtype=auto` follows the model's bf16/fp16
  // compute dtype, not the GPU's FP8 capability. FP8 KV is opt-in ("auto").
  kvCachePrecision: "fp16",
};

interface GpuSetupStats {
  maxConcurrentStreams: number;
  decodeThroughputTokS: number | null;
}

/**
 * Calculate the maximum concurrent streams a GPU setup can handle,
 * plus raw single-stream decode throughput.
 *
 * The memoryUtilization parameter (0–1) mirrors vLLM's --gpu-memory-utilization:
 * it caps the usable fraction of total VRAM for weights + KV cache.
 * Delegates to shared functions in calculations.ts for throughput and KV cache.
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
  const precision = resolveModelPrecision(model);

  // Decode throughput via shared function (raw single-stream)
  const decodeThroughput = calcDecodeThroughput(
    model, precision, gpuName, gpuCount, interconnect,
  );
  if (decodeThroughput === null) return { maxConcurrentStreams: 0, decodeThroughputTokS: null };

  // Resolve KV cache precision from settings. Default "fp16" (2 B) mirrors
  // vLLM's `kv_cache_dtype=auto`, which tracks the model's bf16/fp16 compute
  // dtype; "auto" here opts into FP8 (1 B) on FP8-KV-capable GPUs.
  const kvPrecisionBytes = resolveKvPrecisionBytes(settings.kvCachePrecision, gpuName);

  // Apply memory utilization to total VRAM (like vLLM's --gpu-memory-utilization)
  const usableVramGb = totalVramGb * memoryUtilization;

  // Max concurrent requests via shared function (physics-based VRAM budget)
  const maxConcurrentStreams = calcMaxConcurrentRequests(
    model, precision, usableVramGb,
    settings.avgInputTokens, settings.avgOutputTokens,
    kvPrecisionBytes,
  );

  return {
    maxConcurrentStreams,
    decodeThroughputTokS: decodeThroughput,
  };
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
      });
      break;
    }
  }

  options.sort((a, b) => a.monthlyCost - b.monthlyCost);
  return options.slice(0, limit);
}

/**
 * Calculate the recommendation matrix for the Performance persona.
 *
 * Returns a 2D array: rows = models (sorted by score desc), columns = concurrency tiers.
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

    return CONCURRENCY_TIERS.map((tier: ConcurrencyTierConfig): MatrixCell => {
      let gpuSetups = findGpuSetups(
        model,
        allGpus,
        tier.midpoint,
        settings,
      );

      // Performance persona: scale up if no real offering suffices
      if (gpuSetups.length === 0) {
        gpuSetups = findScaledGpuSetups(model, allGpus, tier.midpoint, settings);
      }

      const cheapest = gpuSetups.length > 0 ? gpuSetups[0] : null;

      return {
        model,
        benchmark,
        sotaScore: sota,
        percentOfSota,
        totalBenchmarkCost,
        gpuSetups,
        costPerStreamPerMonth: cheapest?.costPerStreamPerMonth ?? null,
        exceedsCapacity: gpuSetups.length === 0,
        decodeThroughputTokS: cheapest?.decodeThroughputTokS ?? null,
        utilization: cheapest
          ? Math.min(tier.midpoint / cheapest.maxConcurrentStreams, 1.0)
          : null,
        isUnranked: false,
      };
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
    CONCURRENCY_TIERS.map((tier: ConcurrencyTierConfig): MatrixCell => {
      let gpuSetups = findGpuSetups(model, allGpus, tier.midpoint, settings);
      if (gpuSetups.length === 0) {
        gpuSetups = findScaledGpuSetups(model, allGpus, tier.midpoint, settings);
      }
      const cheapest = gpuSetups.length > 0 ? gpuSetups[0] : null;

      return {
        model,
        benchmark: null,
        sotaScore: null,
        percentOfSota: null,
        totalBenchmarkCost: null,
        gpuSetups,
        costPerStreamPerMonth: cheapest?.costPerStreamPerMonth ?? null,
        exceedsCapacity: gpuSetups.length === 0,
        decodeThroughputTokS: cheapest?.decodeThroughputTokS ?? null,
        utilization: cheapest
          ? Math.min(tier.midpoint / cheapest.maxConcurrentStreams, 1.0)
          : null,
        isUnranked: true,
      };
    }),
  );
}
