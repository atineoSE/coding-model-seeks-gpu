import type { ApiPricingEntry, Model, GpuOffering, AdvancedSettings } from "@/types";
import { findGpuSetups } from "./matrix-calculator";

export const COMPACTION_THRESHOLD_TOKENS = 20_000;

export const PROVIDER_CACHE_TTLS: Record<string, number[]> = {
  anthropic: [5, 60, 1440],
  openai: [60],
  google: [],
};

export interface CostConfig {
  turnsPerConversation: number;
  cacheHitRate: number;        // 0.0–1.0
  cacheTtlMin: number | null;  // null for providers without TTL
  avgInputTokens: number;
  avgOutputTokens: number;
}

// Assumed gap between conversations in minutes (for cache expiry check on turn 1)
const INTER_CONVERSATION_GAP_MIN = 60;

export function computeAvgCostPerTurn(
  entry: ApiPricingEntry,
  config: CostConfig,
): number {
  const { turnsPerConversation: T, cacheHitRate: c, cacheTtlMin, avgInputTokens: I, avgOutputTokens: O } = config;

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

    // Turn 1: cache may have expired if TTL shorter than inter-conversation gap
    const effectiveHitRate =
      t === 1 && (cacheTtlMin == null || cacheTtlMin < INTER_CONVERSATION_GAP_MIN)
        ? 0
        : c;

    const cached = totalInput * effectiveHitRate;
    const fresh = totalInput - cached;

    const turnCost = hasCaching
      ? fresh * inputCost + cached * cacheReadCost + O * cacheWriteCost + O * outputCost
      : totalInput * inputCost + O * outputCost;

    totalCost += turnCost;
  }

  return totalCost / T;
}

export function computeApiCostPoints(
  entry: ApiPricingEntry,
  config: CostConfig,
  maxTurns: number,
  steps = 200,
): { x: number; y: number }[] {
  const avgCostPerTurn = computeAvgCostPerTurn(entry, config);
  return Array.from({ length: steps + 1 }, (_, i) => {
    const x = (maxTurns / steps) * i;
    return { x, y: x * avgCostPerTurn };
  });
}

export function computeSelfHostingMonthlyCost(
  model: Model,
  gpus: GpuOffering[],
  settings: AdvancedSettings,
): number | null {
  const setups = findGpuSetups(model, gpus, 1, settings);
  return setups.length > 0 ? setups[0].monthlyCost : null;
}

export function findIntersection(
  avgCostPerTurn: number,
  flatCost: number,
): number | null {
  if (avgCostPerTurn <= 0) return null;
  return flatCost / avgCostPerTurn;
}

export function getProviderCacheTtls(entry: ApiPricingEntry): number[] {
  return PROVIDER_CACHE_TTLS[entry.lab] ?? [];
}
