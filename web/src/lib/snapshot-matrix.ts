/**
 * Logic for the Snapshot Coverage Matrix — determines which models to show
 * and how to sort them.
 *
 * Mirrors the Python `find_best_models_per_lab` from
 * `pipeline/pipeline/sources/litellm_source.py`.
 */

import type { BenchmarkScore, Model } from "@/types";

// Prefix-based lab detection — mirrors LAB_PATTERNS in
// pipeline/pipeline/sources/litellm_source.py. New models are picked up
// automatically as long as they follow the naming convention.
export const LAB_PATTERNS: [prefix: string, lab: string][] = [
  ["claude-", "anthropic"],
  ["gpt-", "openai"],
  ["gemini-", "google"],
];

export function getLabForModel(modelName: string): string | null {
  const lower = modelName.toLowerCase();
  for (const [prefix, lab] of LAB_PATTERNS) {
    if (lower.startsWith(prefix)) return lab;
  }
  return null;
}

/** The 5 benchmark categories (excluding the derived "overall"). */
export const BENCHMARK_CATEGORIES = [
  "frontend",
  "greenfield",
  "issue_resolution",
  "testing",
  "information_gathering",
] as const;

export type BenchmarkCategory = (typeof BENCHMARK_CATEGORIES)[number];

export const CATEGORY_DISPLAY_NAMES: Record<BenchmarkCategory, string> = {
  frontend: "Frontend",
  greenfield: "Greenfield",
  issue_resolution: "Issue Resolution",
  testing: "Testing",
  information_gathering: "Information Gathering",
};

export interface MatrixModel {
  modelName: string;
  lab: string | null; // null for open-weights models
  overallScore: number | null;
  scores: Record<string, number | null>; // benchmark_name → score
  // True when the model has no usable benchmark score in the snapshot — a
  // sized open-weights model that has not landed on the OpenHands Index yet
  // (partial-model-data skill). Closed best-in-lab models are never unranked.
  unranked: boolean;
}

/**
 * Find the best closed model per lab — TypeScript port of Python's
 * `find_best_models_per_lab`.
 */
export function findBestModelsPerLab(
  benchmarks: BenchmarkScore[],
): Record<string, string> {
  const best: Record<string, { hasOverall: boolean; score: number; modelName: string }> = {};

  for (const entry of benchmarks) {
    const modelName = entry.model_name;
    const lab = getLabForModel(modelName);
    if (lab === null) continue;
    if (entry.openness !== "closed_api_available") continue;

    const score = entry.score;
    if (score == null) continue;

    const isOverall = entry.benchmark_name === "overall";
    const current = best[lab];

    if (!current) {
      best[lab] = { hasOverall: isOverall, score, modelName };
    } else {
      if ((!current.hasOverall && isOverall) || (current.hasOverall === isOverall && score > current.score)) {
        best[lab] = { hasOverall: isOverall, score, modelName };
      }
    }
  }

  const result: Record<string, string> = {};
  for (const [lab, { modelName }] of Object.entries(best)) {
    result[lab] = modelName;
  }
  return result;
}

/**
 * Build the list of models to display in the snapshot matrix.
 *
 * - All open-weights models that appear in the snapshot (ranked)
 * - Open-weights models from `models` (models.json) that have no snapshot
 *   benchmark entry yet (unranked — sized, no OpenHands Index result)
 * - Best closed model per lab (Anthropic, OpenAI, Google)
 *
 * Sorted by overall score descending; models without a usable score (unranked)
 * go to the bottom. Each model carries an `unranked` flag so the caller can
 * group ranked vs unranked open-weights models separately.
 */
export function getMatrixModels(
  benchmarks: BenchmarkScore[],
  models: Model[] = [],
): MatrixModel[] {
  const bestPerLab = findBestModelsPerLab(benchmarks);
  const selectedClosedModels = new Set(Object.values(bestPerLab));

  // Invert bestPerLab for lookup: model → lab
  const modelToLab: Record<string, string> = {};
  for (const [lab, model] of Object.entries(bestPerLab)) {
    modelToLab[model] = lab;
  }

  // Collect unique model names that qualify
  const qualifiedModels = new Set<string>();
  for (const entry of benchmarks) {
    if (entry.openness === "open_weights") {
      qualifiedModels.add(entry.model_name);
    } else if (selectedClosedModels.has(entry.model_name)) {
      qualifiedModels.add(entry.model_name);
    }
  }

  // Add open-weights models from models.json. These are all self-hostable open
  // models (closed models are never in models.json); any that have no snapshot
  // benchmark entry surface here as unranked (their scores stay null).
  for (const m of models) {
    qualifiedModels.add(m.model_name);
  }

  // Build MatrixModel for each qualified model
  const modelMap = new Map<string, MatrixModel>();
  for (const name of qualifiedModels) {
    modelMap.set(name, {
      modelName: name,
      lab: modelToLab[name] ?? null,
      overallScore: null,
      scores: {},
      unranked: false,
    });
  }

  for (const entry of benchmarks) {
    const model = modelMap.get(entry.model_name);
    if (!model) continue;

    if (entry.benchmark_name === "overall") {
      model.overallScore = entry.score;
    }
    model.scores[entry.benchmark_name] = entry.score;
  }

  // A model is unranked when it has no usable score anywhere in the snapshot —
  // no `overall` and no non-null category score.
  for (const model of modelMap.values()) {
    const hasScore =
      model.overallScore != null ||
      Object.values(model.scores).some((v) => v != null);
    model.unranked = !hasScore;
  }

  const avgScore = (m: MatrixModel): number | null => {
    const vals = Object.entries(m.scores)
      .filter(([k]) => k !== "overall")
      .map(([, v]) => v)
      .filter((v): v is number => v != null);
    return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  // Sort by overall score if available, otherwise by average of available scores
  const matrixModels = Array.from(modelMap.values());
  matrixModels.sort((a, b) => {
    const scoreA = a.overallScore ?? avgScore(a);
    const scoreB = b.overallScore ?? avgScore(b);
    if (scoreA != null && scoreB != null) return scoreB - scoreA;
    if (scoreA != null) return -1;
    if (scoreB != null) return 1;
    return a.modelName.localeCompare(b.modelName);
  });

  return matrixModels;
}
