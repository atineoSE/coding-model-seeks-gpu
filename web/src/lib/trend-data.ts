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

export function computeGapTrend(
  snapshots: SnapshotData[],
  openSourceNames: Set<string>,
  category: string,
): GapTrendPoint[] {
  const points: GapTrendPoint[] = [];

  for (const snap of snapshots) {
    const categoryBenchmarks = snap.benchmarks.filter(
      (b) => b.benchmark_name === category && b.score !== null,
    );
    if (categoryBenchmarks.length === 0) continue;

    let bestClosed: { score: number; model: string } | null = null;
    let bestOpen: { score: number; model: string } | null = null;

    for (const b of categoryBenchmarks) {
      const resolved = resolveModelName(b.model_name);
      const isOpen = openSourceNames.has(resolved);

      if (isOpen) {
        if (!bestOpen || b.score! > bestOpen.score) {
          bestOpen = { score: b.score!, model: resolved };
        }
      } else {
        if (!bestClosed || b.score! > bestClosed.score) {
          bestClosed = { score: b.score!, model: resolved };
        }
      }
    }

    if (bestClosed && bestOpen) {
      points.push({
        date: snap.date,
        closedSourceScore: bestClosed.score,
        closedSourceModel: bestClosed.model,
        openSourceScore: bestOpen.score,
        openSourceModel: bestOpen.model,
      });
    }
  }

  return points;
}

/**
 * Keep only change-points: the first point plus any point where the score
 * or model name differs from its predecessor.
 */
export function deduplicateGapTrend(
  points: GapTrendPoint[],
): GapTrendPoint[] {
  if (points.length <= 1) return points;

  const result: GapTrendPoint[] = [points[0]];

  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    if (
      curr.closedSourceScore !== prev.closedSourceScore ||
      curr.closedSourceModel !== prev.closedSourceModel ||
      curr.openSourceScore !== prev.openSourceScore ||
      curr.openSourceModel !== prev.openSourceModel
    ) {
      result.push(curr);
    }
  }

  return result;
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

export function computeCostTrend(
  snapshots: SnapshotData[],
  openSourceNames: Set<string>,
  models: Model[],
  gpus: GpuOffering[],
  category: string,
  settings: AdvancedSettings,
): CostTrendPoint[] {
  const points: CostTrendPoint[] = [];
  const concurrency = 5; // multi_agent midpoint

  for (const snap of snapshots) {
    const categoryBenchmarks = snap.benchmarks.filter(
      (b) => b.benchmark_name === category && b.score !== null,
    );

    // Find best open-source model in this snapshot
    let bestOpen: { name: string; score: number } | null = null;
    for (const b of categoryBenchmarks) {
      const resolved = resolveModelName(b.model_name);
      if (!openSourceNames.has(resolved)) continue;
      if (!bestOpen || b.score! > bestOpen.score) {
        bestOpen = { name: resolved, score: b.score! };
      }
    }
    if (!bestOpen) continue;

    // Look up the Model object
    const model = models.find((m) => m.model_name === bestOpen!.name);
    if (!model) continue;

    // Find cheapest GPU setup at concurrency=5
    let setups = findGpuSetups(model, gpus, concurrency, settings);
    if (setups.length === 0) {
      setups = findScaledGpuSetups(model, gpus, concurrency, settings);
    }
    if (setups.length === 0) continue;

    const cheapest = setups[0];
    const memGb = getModelMemory(model, resolveModelPrecision(model));
    points.push({
      date: snap.date,
      monthlyCost: cheapest.monthlyCost,
      modelName: bestOpen.name,
      gpuSetup: `${cheapest.gpuCount}× ${cheapest.gpuName}`,
      score: bestOpen.score,
      modelMemoryGb: memGb ?? 0,
    });
  }

  return points;
}

/**
 * Keep only points where the best open-source model changes.
 * First point always kept.
 */
export function deduplicateCostTrend(
  points: CostTrendPoint[],
): CostTrendPoint[] {
  if (points.length <= 1) return points;

  const result: CostTrendPoint[] = [points[0]];

  for (let i = 1; i < points.length; i++) {
    if (points[i].modelName !== points[i - 1].modelName) {
      result.push(points[i]);
    }
  }

  return result;
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
