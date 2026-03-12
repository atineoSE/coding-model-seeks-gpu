import type { BenchmarkScore } from "@/types";

/**
 * Canonical instance counts per benchmark category.
 * Used to compute total benchmark cost = cost_per_task × instance_count.
 */
export const BENCHMARK_INSTANCE_COUNTS: Record<string, number> = {
  issue_resolution: 500,
  frontend: 102,
  greenfield: 16,
  testing: 433,
  information_gathering: 165,
};

const ALL_CATEGORIES = Object.keys(BENCHMARK_INSTANCE_COUNTS);

/**
 * Compute the total API cost to run a full benchmark for a given model.
 *
 * For a specific category: cost_per_task × instanceCount.
 * For "overall": sum of all 5 categories' (cost_per_task × instanceCount).
 *
 * Returns null if any required cost data is missing.
 */
export function computeTotalBenchmarkCost(
  modelName: string,
  benchmarkCategory: string,
  allBenchmarks: BenchmarkScore[],
): number | null {
  if (benchmarkCategory === "overall") {
    let total = 0;
    for (const cat of ALL_CATEGORIES) {
      const entry = allBenchmarks.find(
        (b) => b.model_name === modelName && b.benchmark_name === cat,
      );
      if (!entry || entry.cost_per_task === null) return null;
      total += entry.cost_per_task * BENCHMARK_INSTANCE_COUNTS[cat];
    }
    return total;
  }

  const instanceCount = BENCHMARK_INSTANCE_COUNTS[benchmarkCategory];
  if (instanceCount === undefined) return null;

  const entry = allBenchmarks.find(
    (b) => b.model_name === modelName && b.benchmark_name === benchmarkCategory,
  );
  if (!entry || entry.cost_per_task === null) return null;

  return entry.cost_per_task * instanceCount;
}
