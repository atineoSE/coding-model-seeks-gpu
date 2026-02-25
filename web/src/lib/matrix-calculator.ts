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


const HOURS_PER_MONTH = 720;

/**
 * KV memory safety margin (0–1).
 *
 * PagedAttention block fragmentation, memory allocator overhead, and runtime
 * CUDA state mean the theoretical max KV slots are never fully usable.
 * 0.75 = use at most 75% of the theoretical KV budget.
 */
const KV_MEMORY_SAFETY_MARGIN = 0.75;

export const DEFAULT_ADVANCED_SETTINGS: AdvancedSettings = {
  avgInputTokens: 4000,
  avgOutputTokens: 1500,
  minTokPerStream: 20,
  prefixCacheHitRate: 80,
};

interface GpuSetupStats {
  maxConcurrentStreams: number;
  decodeThroughputTokS: number | null;
}

/**
 * Calculate the maximum concurrent streams a GPU setup can handle,
 * plus raw single-stream decode throughput.
 * Delegates to shared functions in calculations.ts for throughput and KV cache.
 */
export function calcGpuSetupStats(
  model: Model,
  gpuName: string,
  gpuCount: number,
  totalVramGb: number,
  interconnect: string | null,
  settings: AdvancedSettings,
): GpuSetupStats {
  const precision = resolveModelPrecision(model);

  // Decode throughput via shared function (raw single-stream)
  const decodeThroughput = calcDecodeThroughput(
    model, precision, gpuName, gpuCount, interconnect,
  );
  if (decodeThroughput === null) return { maxConcurrentStreams: 0, decodeThroughputTokS: null };

  // Resolve KV cache precision — always auto (FP8 on supported GPUs)
  const kvPrecisionBytes = resolveKvPrecisionBytes("auto", gpuName);

  // Convert prefix cache hit rate (%) to cache utilization (80% → 0.20)
  const avgCacheUtilization = 1 - settings.prefixCacheHitRate / 100;

  // Max concurrent requests via shared function (physics-based VRAM budget)
  const kvMax = calcMaxConcurrentRequests(
    model, precision, totalVramGb,
    settings.avgInputTokens, settings.avgOutputTokens,
    kvPrecisionBytes, avgCacheUtilization,
  );
  if (kvMax === 0) return { maxConcurrentStreams: 0, decodeThroughputTokS: decodeThroughput };

  // Apply KV memory safety margin for fragmentation / allocator overhead
  const maxConcurrentStreams = Math.floor(kvMax * KV_MEMORY_SAFETY_MARGIN);

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
        gpuSetups,
        costPerStreamPerMonth: cheapest?.costPerStreamPerMonth ?? null,
        exceedsCapacity: gpuSetups.length === 0,
        decodeThroughputTokS: cheapest?.decodeThroughputTokS ?? null,
        utilization: cheapest
          ? Math.min(tier.midpoint / cheapest.maxConcurrentStreams, 1.0)
          : null,
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
        gpuSetups: gpuSetup ? [gpuSetup] : [],
        costPerStreamPerMonth: costPerStream,
        exceedsCapacity: cannotServe,
        decodeThroughputTokS: cannotServe ? null : stats.decodeThroughputTokS,
        utilization: stats.maxConcurrentStreams > 0
          ? Math.min(tier.midpoint / stats.maxConcurrentStreams, 1.0)
          : null,
      };
    });
  });
}

// ============================================================================
// Budget Chart Data (team-size view)
// ============================================================================

export interface BudgetChartDataPoint {
  modelName: string;
  concurrentStreams: number;
  teamSizeIde: number;
  teamSizeCli: number;
  teamSizeAvg: number;
  percentOfSota: number;
  modelMemoryGb: number;
  fits: boolean;
  doesntFitReason: string | null;
  decodeThroughputTokS: number | null;
  maxConcurrentStreams: number;
  benchmarkScore: number | null;
}

/**
 * Calculate budget chart data for the team-size view.
 *
 * For each ranked model (by benchmark score), compute how many concurrent
 * streams the GPU config can serve and translate that into development
 * team sizes for IDE-workflow and CLI-workflow patterns.
 */
export function calculateBudgetChartData(
  gpuConfig: PresetGpuConfig,
  allModels: Model[],
  benchmarks: BenchmarkScore[],
  sotaScores: SotaScore[],
  benchmarkCategory: string,
  targetUtilization: number,
  minTokPerSec: number,
  ideStreamsPerDev: number,
  cliStreamsPerDev: number,
  settings: AdvancedSettings,
): BudgetChartDataPoint[] {
  // Filter benchmarks for the selected category
  const categoryBenchmarks = benchmarks.filter(
    (b) => b.benchmark_name === benchmarkCategory,
  );

  // Find SOTA score for the category
  const sota = sotaScores.find((s) => s.benchmark_name === benchmarkCategory) ?? null;

  // Match benchmarks to models and sort by score descending
  const modelScores: { model: Model; benchmark: BenchmarkScore }[] = [];
  for (const b of categoryBenchmarks) {
    if (b.score === null) continue;
    const model = allModels.find((m) => m.model_name === b.model_name);
    if (!model) continue;
    modelScores.push({ model, benchmark: b });
  }
  modelScores.sort((a, b) => (b.benchmark.score ?? 0) - (a.benchmark.score ?? 0));

  return modelScores.map(({ model, benchmark }): BudgetChartDataPoint => {
    const precision = resolveModelPrecision(model);
    const modelMemoryGb = getModelMemory(model, precision);

    // Check if model physically fits
    const physicallyFits = modelMemoryGb !== null
      ? gpusNeeded(modelMemoryGb * WEIGHT_OVERHEAD_FACTOR, gpuConfig.vramPerGpu) <= gpuConfig.gpuCount
      : false;

    const percentOfSota =
      sota && benchmark.score !== null && sota.sota_score > 0
        ? (benchmark.score / sota.sota_score) * 100
        : 0;

    if (!physicallyFits || modelMemoryGb === null) {
      return {
        modelName: model.model_name,
        concurrentStreams: 0,
        teamSizeIde: 0,
        teamSizeCli: 0,
        teamSizeAvg: 0,
        percentOfSota,
        modelMemoryGb: modelMemoryGb ?? 0,
        fits: false,
        doesntFitReason: "Not enough VRAM",
        decodeThroughputTokS: null,
        maxConcurrentStreams: 0,
        benchmarkScore: benchmark.score,
      };
    }

    // Compute stats
    const stats = calcGpuSetupStats(
      model,
      gpuConfig.gpuName,
      gpuConfig.gpuCount,
      gpuConfig.totalVramGb,
      gpuConfig.interconnect,
      settings,
    );

    const throughputTooLow =
      stats.decodeThroughputTokS !== null &&
      stats.decodeThroughputTokS < minTokPerSec;

    const fits = stats.maxConcurrentStreams > 0 && !throughputTooLow;

    let doesntFitReason: string | null = null;
    if (!fits) {
      if (stats.maxConcurrentStreams === 0) {
        doesntFitReason = "Not enough VRAM for KV cache";
      } else if (throughputTooLow) {
        doesntFitReason = `Throughput too low (${Math.round(stats.decodeThroughputTokS!)} tok/s < ${minTokPerSec} min)`;
      }
    }

    const concurrentStreams = fits
      ? Math.floor(stats.maxConcurrentStreams * targetUtilization)
      : 0;

    const teamSizeIde = concurrentStreams / ideStreamsPerDev;
    const teamSizeCli = concurrentStreams / cliStreamsPerDev;
    const teamSizeAvg = (teamSizeIde + teamSizeCli) / 2;

    return {
      modelName: model.model_name,
      concurrentStreams,
      teamSizeIde,
      teamSizeCli,
      teamSizeAvg,
      percentOfSota,
      modelMemoryGb,
      fits,
      doesntFitReason,
      decodeThroughputTokS: stats.decodeThroughputTokS,
      maxConcurrentStreams: stats.maxConcurrentStreams,
      benchmarkScore: benchmark.score,
    };
  });
}
