/**
 * Usage Regime Parameters
 *
 * Defines three distinct usage patterns for LLM inference:
 * - Low-concurrency: Small teams with few concurrent requests
 * - High-concurrency: Large teams or production APIs with many concurrent requests
 * - Long-context: Few requests but very large context windows
 */

export type UsageRegime = "low-concurrency" | "high-concurrency" | "long-context";

export interface RegimeParameters {
  /** Typical number of concurrent requests */
  concurrentRequests: number;

  /** Average input tokens per request (0 = use model's context_length * 0.5 for long-context) */
  avgInputTokens: number;

  /** Average output tokens per request */
  avgOutputTokens: number;

  /** Prompts per user per hour (request rate) */
  promptsPerUserPerHour: number;

  /** Burst factor (peak load multiplier over average) */
  burstFactor: number;

  /** Target utilization (comfort margin) */
  utilizationTarget: number;

  /** Human-readable description */
  description: string;

  /** Example use cases */
  examples: string[];
}

/**
 * Regime parameter definitions.
 *
 * These are informed by:
 * - vLLM performance benchmarks
 * - Real-world production deployments
 * - KV cache memory profiling data
 */
export const REGIME_PARAMS: Record<UsageRegime, RegimeParameters> = {
  "low-concurrency": {
    concurrentRequests: 10,
    avgInputTokens: 1000,
    avgOutputTokens: 500,
    promptsPerUserPerHour: 20,
    burstFactor: 2.0,
    utilizationTarget: 0.75,     // Run at 75% utilization for comfort
    description: "Small teams with occasional requests and quick responses",
    examples: [
      "Engineering team using Copilot",
      "Research team with Q&A assistant",
      "Small company internal chatbot",
    ],
  },

  "high-concurrency": {
    concurrentRequests: 100,
    avgInputTokens: 500,
    avgOutputTokens: 200,
    promptsPerUserPerHour: 30,
    burstFactor: 2.5,
    utilizationTarget: 0.70,     // Lower target due to higher variability
    description: "Large teams or production APIs with many concurrent users",
    examples: [
      "Customer-facing chatbot",
      "Large enterprise internal assistant",
      "API service with hundreds of users",
    ],
  },

  "long-context": {
    concurrentRequests: 5,
    avgInputTokens: 0,  // Will use model.context_length * 0.5 as input
    avgOutputTokens: 2000,
    promptsPerUserPerHour: 10,
    burstFactor: 1.5,
    utilizationTarget: 0.75,
    description: "Few requests with very large context windows (RAG, document analysis)",
    examples: [
      "Document summarization pipeline",
      "Legal contract analysis",
      "RAG with large knowledge bases",
    ],
  },
};

/**
 * Get regime parameters by name.
 */
export function getRegimeParams(regime: UsageRegime): RegimeParameters {
  return REGIME_PARAMS[regime];
}

/**
 * Get all regime names.
 */
export function getRegimeNames(): UsageRegime[] {
  return Object.keys(REGIME_PARAMS) as UsageRegime[];
}

/**
 * Get display name for regime.
 */
export function getRegimeDisplayName(regime: UsageRegime): string {
  const displayNames: Record<UsageRegime, string> = {
    "low-concurrency": "Low Concurrency",
    "high-concurrency": "High Concurrency",
    "long-context": "Long Context",
  };
  return displayNames[regime];
}
