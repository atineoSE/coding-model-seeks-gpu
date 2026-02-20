import { useState, useEffect } from "react";
import type {
  BenchmarkScore,
  SotaScore,
  Model,
  GpuOffering,
  AdvancedSettings,
  GpuSetupOption,
} from "@/types";
import { findGpuSetups, findScaledGpuSetups } from "./matrix-calculator";
import { getModelMemory, resolveModelPrecision } from "./calculations";
import { GPU_PRESETS } from "./gpu-presets";

// ---------------------------------------------------------------------------
// Model alias map — ported from pipeline/pipeline/snapshots/alias_map.py
// Ignoring dates: always resolve to the current canonical name.
// ---------------------------------------------------------------------------

const MODEL_ALIASES: Record<string, string> = {
  // Early snapshot model names (pre-Jan-15 naming convention)
  "Qwen3-Coder-480B-A35B-Instruct-FP8": "Qwen3-Coder-480B",
  "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
  // 2026-02-10: Bulk rename to official marketing names
  "claude-opus-4-5-20251101": "claude-opus-4-5",
  "claude-4.5-opus": "claude-opus-4-5",
  "claude-4.5-sonnet": "claude-sonnet-4-5",
  "claude-4.6-opus": "claude-opus-4-6",
  "glm-4.7": "GLM-4.7",
  "gpt-5": "GPT-5.2",
  "gpt-5.2": "GPT-5.2",
  "gpt-5.2-codex": "GPT-5.2-Codex",
  "gpt-5.2-high-reasoning": "GPT-5.2-Codex",
  "kimi-k2-thinking": "Kimi-K2-Thinking",
  "kimi-k2.5": "Kimi-K2.5",
  "minimax-m2": "MiniMax-M2.1",
  "minimax-m2.1": "MiniMax-M2.1",
  "nemotron": "Nemotron-3-Nano",
  "nemotron-3-nano": "Nemotron-3-Nano",
  "nemotron-3-nano-30b": "Nemotron-3-Nano",
  "deepseek-v3.2-reasoner": "DeepSeek-V3.2-Reasoner",
  "gemini-3-flash": "Gemini-3-Flash",
  "gemini-3-pro": "Gemini-3-Pro",
  "gemini-3-pro-preview": "Gemini-3-Pro",
  "qwen-3-coder": "Qwen3-Coder-480B",
  // 2026-02-11: jade-spark-2862 → Minimax-2.5
  "jade-spark-2862": "MiniMax-M2.5",
  // 2026-02-12: Minimax-2.5 → MiniMax-M2.5
  "Minimax-2.5": "MiniMax-M2.5",
};

// ---------------------------------------------------------------------------
// Name resolution helpers
// ---------------------------------------------------------------------------

/** Apply alias map to normalize model names across snapshots. */
export function resolveModelName(name: string): string {
  let current = name;
  const seen = new Set<string>([current]);
  while (MODEL_ALIASES[current]) {
    current = MODEL_ALIASES[current];
    if (seen.has(current)) break;
    seen.add(current);
  }
  return current;
}

/** A model is open-source if its resolved name matches any entry in models.json. */
export function isOpenSourceModel(
  name: string,
  openSourceNames: Set<string>,
): boolean {
  return openSourceNames.has(resolveModelName(name));
}

// ---------------------------------------------------------------------------
// Snapshot data types & loading
// ---------------------------------------------------------------------------

export interface SnapshotData {
  date: string;
  benchmarks: BenchmarkScore[];
  sotaScores: SotaScore[];
}

interface SnapshotIndex {
  snapshots: string[];
  latest: string | null;
}

export function useSnapshotData(): {
  snapshots: SnapshotData[];
  loading: boolean;
} {
  const [snapshots, setSnapshots] = useState<SnapshotData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const index: SnapshotIndex = await fetch("/data/snapshots/index.json").then(
          (r) => r.json(),
        );

        const results = await Promise.all(
          index.snapshots.map(async (date) => {
            const base = `/data/snapshots/${date}`;
            const [benchmarks, sotaScores] = await Promise.all([
              fetch(`${base}/benchmarks.json`).then((r) => r.json()),
              fetch(`${base}/sota_scores.json`)
                .then((r) => r.json())
                .catch(() => []),
            ]);
            return { date, benchmarks, sotaScores } as SnapshotData;
          }),
        );

        if (!cancelled) {
          setSnapshots(results);
        }
      } catch {
        if (!cancelled) {
          setSnapshots([]);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return { snapshots, loading };
}

// ---------------------------------------------------------------------------
// Chart 1: Gap Trend
// ---------------------------------------------------------------------------

export interface GapTrendPoint {
  date: string;
  closedSourceScore: number;
  closedSourceModel: string;
  openSourceScore: number;
  openSourceModel: string;
}

/**
 * Build a lookup of latest scores per model from the most recent snapshot.
 * This is the single source of truth for all score evaluations.
 */
function buildLatestScores(
  snapshots: SnapshotData[],
  category: string,
): Map<string, number> {
  const scores = new Map<string, number>();
  if (snapshots.length === 0) return scores;

  const latest = snapshots[snapshots.length - 1];
  for (const b of latest.benchmarks) {
    if (b.benchmark_name !== category || b.score === null) continue;
    const resolved = resolveModelName(b.model_name);
    const existing = scores.get(resolved);
    if (existing === undefined || b.score! > existing) {
      scores.set(resolved, b.score!);
    }
  }
  return scores;
}

/**
 * Build first-appearance dates per model by scanning snapshots chronologically.
 */
function buildFirstSeen(
  snapshots: SnapshotData[],
  category: string,
): Map<string, string> {
  const firstSeen = new Map<string, string>();
  for (const snap of snapshots) {
    for (const b of snap.benchmarks) {
      if (b.benchmark_name !== category || b.score === null) continue;
      const resolved = resolveModelName(b.model_name);
      if (!firstSeen.has(resolved)) {
        firstSeen.set(resolved, snap.date);
      }
    }
  }
  return firstSeen;
}

/**
 * Roster-based gap trend: emit a point only when a new model appears and
 * displaces a leader. Uses latest scores for all evaluations.
 */
export function computeGapTrend(
  snapshots: SnapshotData[],
  openSourceNames: Set<string>,
  category: string,
): GapTrendPoint[] {
  if (snapshots.length === 0) return [];

  const latestScores = buildLatestScores(snapshots, category);
  const firstSeen = buildFirstSeen(snapshots, category);
  const points: GapTrendPoint[] = [];

  const roster = new Set<string>();
  let prevClosedModel: string | null = null;
  let prevOpenModel: string | null = null;

  for (const snap of snapshots) {
    // Expand roster with models appearing in this snapshot
    const newModels: string[] = [];
    for (const b of snap.benchmarks) {
      if (b.benchmark_name !== category || b.score === null) continue;
      const resolved = resolveModelName(b.model_name);
      if (!roster.has(resolved)) {
        roster.add(resolved);
        newModels.push(resolved);
      }
    }

    // Skip snapshots with no new models (except the first snapshot)
    if (newModels.length === 0 && points.length > 0) continue;

    // Evaluate leaders from the current roster using latest scores
    let bestClosed: { score: number; model: string } | null = null;
    let bestOpen: { score: number; model: string } | null = null;

    for (const model of roster) {
      const score = latestScores.get(model);
      if (score === undefined) continue;

      if (openSourceNames.has(model)) {
        if (!bestOpen || score > bestOpen.score) {
          bestOpen = { score, model };
        }
      } else {
        if (!bestClosed || score > bestClosed.score) {
          bestClosed = { score, model };
        }
      }
    }

    if (!bestClosed || !bestOpen) continue;

    // Emit only if a leader changed (or this is the first point)
    if (
      bestClosed.model !== prevClosedModel ||
      bestOpen.model !== prevOpenModel
    ) {
      // Date = first-appearance of the newer model of the pair
      const closedFirst = firstSeen.get(bestClosed.model) ?? snap.date;
      const openFirst = firstSeen.get(bestOpen.model) ?? snap.date;
      const pointDate = closedFirst > openFirst ? closedFirst : openFirst;

      points.push({
        date: pointDate,
        closedSourceScore: bestClosed.score,
        closedSourceModel: bestClosed.model,
        openSourceScore: bestOpen.score,
        openSourceModel: bestOpen.model,
      });

      prevClosedModel = bestClosed.model;
      prevOpenModel = bestOpen.model;
    }
  }

  return points;
}

// ---------------------------------------------------------------------------
// Chart 2: Cost Trend
// ---------------------------------------------------------------------------

export interface CostTrendPoint {
  date: string;
  monthlyCost: number;
  modelName: string;
  gpuSetup: string;
  score: number;
  modelMemoryGb: number;
}

/**
 * Compute cost trend from canonical gap trend points.
 * For each gap trend point, look up the open-source model's GPU cost.
 */
export function computeCostTrend(
  gapPoints: GapTrendPoint[],
  models: Model[],
  gpus: GpuOffering[],
  settings: AdvancedSettings,
): CostTrendPoint[] {
  const points: CostTrendPoint[] = [];
  const concurrency = 5; // multi_agent midpoint

  for (const gp of gapPoints) {
    const model = models.find((m) => m.model_name === gp.openSourceModel);
    if (!model) continue;

    let setups = findGpuSetups(model, gpus, concurrency, settings);
    if (setups.length === 0) {
      setups = findScaledGpuSetups(model, gpus, concurrency, settings);
    }
    if (setups.length === 0) continue;

    const cheapest = setups[0];
    const memGb = getModelMemory(model, resolveModelPrecision(model));
    points.push({
      date: gp.date,
      monthlyCost: cheapest.monthlyCost,
      modelName: gp.openSourceModel,
      gpuSetup: `${cheapest.gpuCount}× ${cheapest.gpuName}`,
      score: gp.openSourceScore,
      modelMemoryGb: memGb ?? 0,
    });
  }

  return points;
}

// ---------------------------------------------------------------------------
// Chart 3: Scaling Curve
// ---------------------------------------------------------------------------

export interface ScalingCurvePoint {
  concurrency: number;
  monthlyCost: number;
  gpuSetup: string;
  gpuUtilisation: number; // 0-100%
}

const CONCURRENCY_SAMPLES = [
  1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
  110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
];

export function computeScalingCurve(
  model: Model,
  gpus: GpuOffering[],
  settings: AdvancedSettings,
): ScalingCurvePoint[] {
  const points: ScalingCurvePoint[] = [];

  for (const concurrency of CONCURRENCY_SAMPLES) {
    let setups: GpuSetupOption[] = findGpuSetups(
      model,
      gpus,
      concurrency,
      settings,
    );
    if (setups.length === 0) {
      setups = findScaledGpuSetups(model, gpus, concurrency, settings);
    }
    if (setups.length === 0) continue;

    const cheapest = setups[0];
    const utilisation =
      cheapest.maxConcurrentStreams > 0
        ? Math.round((concurrency / cheapest.maxConcurrentStreams) * 100)
        : 0;
    points.push({
      concurrency,
      monthlyCost: cheapest.monthlyCost,
      gpuSetup: `${cheapest.gpuCount}× ${cheapest.gpuName}`,
      gpuUtilisation: Math.min(utilisation, 100),
    });
  }

  return points;
}

// ---------------------------------------------------------------------------
// GPU Reference Costs (horizontal lines for Charts 2 & 3)
// ---------------------------------------------------------------------------

export interface GpuReferenceCost {
  label: string;
  monthlyCost: number;
}

const REFERENCE_PRESETS = [
  "1× H100 80GB",
  "4× H100 80GB",
  "8× H100 80GB",
  "1× H200 141GB",
  "8× A100 80GB",
];

export function computeGpuReferenceCosts(
  gpus: GpuOffering[],
): GpuReferenceCost[] {
  const costs: GpuReferenceCost[] = [];

  for (const label of REFERENCE_PRESETS) {
    const preset = GPU_PRESETS.find((p) => p.label === label);
    if (!preset) continue;

    const offerings = gpus.filter(
      (g) =>
        g.gpu_name === preset.gpuName &&
        g.gpu_count === preset.gpuCount,
    );
    if (offerings.length === 0) continue;

    const cheapest = offerings.reduce((a, b) =>
      a.price_per_hour < b.price_per_hour ? a : b,
    );

    costs.push({
      label,
      monthlyCost: cheapest.price_per_hour * 720,
    });
  }

  return costs;
}

// ---------------------------------------------------------------------------
// Helper: find best open-source model from latest snapshot
// ---------------------------------------------------------------------------

export function findBestOpenSourceModel(
  benchmarks: BenchmarkScore[],
  openSourceNames: Set<string>,
  models: Model[],
  category: string,
): Model | null {
  const categoryBenchmarks = benchmarks.filter(
    (b) => b.benchmark_name === category && b.score !== null,
  );

  let bestName: string | null = null;
  let bestScore = -Infinity;

  for (const b of categoryBenchmarks) {
    const resolved = resolveModelName(b.model_name);
    if (!openSourceNames.has(resolved)) continue;
    if (b.score! > bestScore) {
      bestScore = b.score!;
      bestName = resolved;
    }
  }

  if (!bestName) return null;
  return models.find((m) => m.model_name === bestName) ?? null;
}
