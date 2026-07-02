import type { ApiPricingEntry, Model, GpuOffering, AdvancedSettings, PresetGpuConfig } from "@/types";
import { findGpuSetups, findScaledGpuSetups, calcGpuSetupStats, requestsPerHourFor } from "./matrix-calculator";

export const COMPACTION_THRESHOLD_TOKENS = 20_000;
export const SECONDS_PER_MONTH = 30 * 24 * 3600;

export const PROVIDER_CACHE_TTLS: Record<string, number[]> = {
  anthropic: [5, 60, 1440],
  openai: [60],
  google: [60],
};

export interface CostConfig {
  requestsPerConversation: number;
  cacheHitRate: number;        // 0.0–1.0
  cacheTtlMin: number | null;  // null for providers without TTL
  avgInputTokens: number;
  avgOutputTokens: number;
}

// Assumed gap between conversations in minutes (for cache expiry check on request 1)
const INTER_CONVERSATION_GAP_MIN = 60;

export function computeAvgCostPerRequest(
  entry: ApiPricingEntry,
  config: CostConfig,
): number {
  const { requestsPerConversation: T, cacheHitRate: c, cacheTtlMin, avgInputTokens: I, avgOutputTokens: O } = config;

  if (!entry.input_cost_per_token || !entry.output_cost_per_token) return 0;

  const contextWindow = entry.context_window ?? Infinity;
  const hasCaching = entry.cache_read_input_token_cost != null;

  const inputCost = entry.input_cost_per_token;
  const outputCost = entry.output_cost_per_token;
  const cacheReadCost = entry.cache_read_input_token_cost ?? 0;
  const cacheWriteCost = entry.cache_creation_input_token_cost ?? 0;

  let totalCost = 0;

  for (let t = 1; t <= T; t++) {
    const raw = COMPACTION_THRESHOLD_TOKENS + (t - 1) * O;
    const ctx = raw > contextWindow ? COMPACTION_THRESHOLD_TOKENS : raw;
    const totalInput = ctx + I;

    // Request 1: cache may have expired if TTL shorter than inter-conversation gap
    const effectiveHitRate =
      t === 1 && (cacheTtlMin == null || cacheTtlMin < INTER_CONVERSATION_GAP_MIN)
        ? 0
        : c;

    const cached = totalInput * effectiveHitRate;
    const fresh = totalInput - cached;

    const requestCost = hasCaching
      ? fresh * inputCost + cached * cacheReadCost + O * cacheWriteCost + O * outputCost
      : totalInput * inputCost + O * outputCost;

    totalCost += requestCost;
  }

  return totalCost / T;
}

export function computeApiCostPoints(
  entry: ApiPricingEntry,
  config: CostConfig,
  maxRequests: number,
  steps = 200,
): { x: number; y: number }[] {
  const avgCostPerRequest = computeAvgCostPerRequest(entry, config);
  return Array.from({ length: steps + 1 }, (_, i) => {
    const x = (maxRequests / steps) * i;
    return { x, y: x * avgCostPerRequest };
  });
}

export function computeSelfHostingMonthlyCost(
  model: Model,
  gpus: GpuOffering[],
  settings: AdvancedSettings,
): number | null {
  const setups = findGpuSetups(model, gpus, 1, settings);
  if (setups.length > 0) return setups[0].monthlyCost;
  const scaled = findScaledGpuSetups(model, gpus, 1, settings);
  return scaled.length > 0 ? scaled[0].monthlyCost : null;
}

export function findIntersection(
  avgCostPerRequest: number,
  flatCost: number,
): number | null {
  if (avgCostPerRequest <= 0) return null;
  return flatCost / avgCostPerRequest;
}

export function getProviderCacheTtls(entry: ApiPricingEntry): number[] {
  return PROVIDER_CACHE_TTLS[entry.lab] ?? [];
}

export function findGpuOfferingForConfig(
  gpuConfig: PresetGpuConfig,
  gpus: GpuOffering[],
): GpuOffering | null {
  const matching = gpus.filter(
    g => g.gpu_name === gpuConfig.gpuName && g.gpu_count === gpuConfig.gpuCount,
  );
  if (matching.length === 0) return null;
  return matching.reduce((a, b) => a.price_per_hour <= b.price_per_hour ? a : b);
}

export interface ConfigSelfHostingCost {
  baseMonthlyCost: number;
  maxRequestsPerMonth: number | null;
}

export function computeSelfHostingCostForConfig(
  model: Model,
  gpuConfig: PresetGpuConfig,
  gpus: GpuOffering[],
  settings: AdvancedSettings,
  memoryUtilization: number,
): ConfigSelfHostingCost | null {
  const offering = findGpuOfferingForConfig(gpuConfig, gpus);
  if (!offering) return null;

  const baseMonthlyCost = offering.price_per_hour * 720;

  const stats = calcGpuSetupStats(
    model,
    gpuConfig.gpuName,
    gpuConfig.gpuCount,
    gpuConfig.totalVramGb,
    gpuConfig.interconnect,
    settings,
    memoryUtilization,
  );

  // Size the box at scale: the aggregate (prefill + batched-decode) throughput,
  // the same number the serving-capacity chart shows — not N × single-stream,
  // which under-serves and makes self-hosting look far pricier than it is.
  const requestsPerHour = requestsPerHourFor(
    stats.deploymentEstimate,
    settings.avgInputTokens,
    settings.avgOutputTokens,
  );
  const maxRequestsPerMonth =
    requestsPerHour !== null ? (requestsPerHour * SECONDS_PER_MONTH) / 3600 : null;

  return { baseMonthlyCost, maxRequestsPerMonth };
}

export function selfHostingStepCost(
  x: number,
  config: ConfigSelfHostingCost,
): number {
  if (config.maxRequestsPerMonth === null) return config.baseMonthlyCost;
  return Math.max(1, Math.ceil(x / config.maxRequestsPerMonth)) * config.baseMonthlyCost;
}

/**
 * Self-hosting cost per request at a monthly volume `x`: the staircase monthly
 * cost amortized over the requests served. Falls as one box fills (B/x), floors
 * at the box's full-utilization cost B/capacity, then steps up when another box
 * is added (a sawtooth). Returns null for x ≤ 0. This is the number to compare
 * against an API's flat per-request price.
 */
export function selfHostingCostPerRequest(
  x: number,
  config: ConfigSelfHostingCost,
): number | null {
  if (x <= 0) return null;
  return selfHostingStepCost(x, config) / x;
}

/**
 * The floor (best-case) self-hosting cost per request — one box's monthly cost
 * over its full capacity (B / capacity). Null when capacity isn't modeled. The
 * crossover with an API price only exists when this floor is below that price.
 */
export function selfHostingFloorCostPerRequest(config: ConfigSelfHostingCost): number | null {
  if (config.maxRequestsPerMonth === null || config.maxRequestsPerMonth <= 0) return null;
  return config.baseMonthlyCost / config.maxRequestsPerMonth;
}
