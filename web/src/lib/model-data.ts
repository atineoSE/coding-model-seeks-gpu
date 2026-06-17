import type { Model, BenchmarkScore } from "@/types";
import {
  getModelMemory,
  resolveModelPrecision,
  WEIGHT_OVERHEAD_FACTOR,
} from "./calculations";

/**
 * Shared model-data helpers.
 *
 * The `partial-model-data` skill requires the model→score mapping to live in a
 * single place so sizing-first views (Budget, Model Size) can treat performance
 * as an optional left join: iterate the MODEL list and ask `scoreFor(...)` per
 * model, rather than iterating benchmarks and dropping models with no score.
 *
 * Terminology (fixed): a model with sizing but no benchmark score is
 * **unranked**. Never coerce a missing score to a number — `null` stays `null`
 * and is rendered as an explicit gap.
 *
 * These helpers are deliberately pure and dependency-light (no React, no data
 * loading) so both the Budget calculators and the Trends charts can import them
 * without circular deps.
 */

/**
 * The benchmark entry for a model in a given category, or `null` if the model
 * is unranked there.
 *
 * Uses the same exact `model_name` join as `matrix-calculator.ts` (no alias
 * resolution). An entry whose `score` is `null` counts as no score, matching
 * the `if (b.score === null) continue` guard the calculators use.
 *
 * Category-specific by design: a model ranked on `overall` but absent from
 * `frontend` is correctly unranked for the `frontend` category.
 */
export function scoreFor(
  model: Model,
  benchmarkCategory: string,
  benchmarks: BenchmarkScore[],
): BenchmarkScore | null {
  return (
    benchmarks.find(
      (b) =>
        b.model_name === model.model_name &&
        b.benchmark_name === benchmarkCategory &&
        b.score !== null,
    ) ?? null
  );
}

/**
 * Whether a model has no benchmark score in the given category — i.e. it is
 * sized (or at least listed) but not yet ranked.
 */
export function isUnranked(
  model: Model,
  benchmarkCategory: string,
  benchmarks: BenchmarkScore[],
): boolean {
  return scoreFor(model, benchmarkCategory, benchmarks) === null;
}

/**
 * Minimum VRAM (GB) needed to serve a model's weights, including activation /
 * CUDA overhead. Mirrors the derivation in `trend-data.ts`'s
 * `computeModelSizeScore` (`getModelMemory(...) * WEIGHT_OVERHEAD_FACTOR`,
 * rounded up to whole GB).
 *
 * Returns `null` when sizing is unknown (e.g. `learnable_params_b` is `null`),
 * so callers can show the gap instead of a fake number.
 */
export function minVramForModel(model: Model): number | null {
  const memGb = getModelMemory(model, resolveModelPrecision(model));
  if (memGb === null) return null;
  return Math.ceil(memGb * WEIGHT_OVERHEAD_FACTOR);
}
