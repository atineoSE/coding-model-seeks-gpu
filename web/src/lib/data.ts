import { useState, useEffect } from "react";
import type { GpuOffering, GpuSource, Model, BenchmarkScore, SotaScore } from "@/types";

export const BASE_PATH = "/coding-model-seeks-gpu";

const DEFAULT_GPU_SOURCE: GpuSource = {
  service_name: "gpuhunt",
  service_url: "https://github.com/dstackai/gpuhunt",
  description: "All regions considered.",
  currency: "USD",
  currency_symbol: "$",
  updated_at: "",
};

export interface AppData {
  gpus: GpuOffering[];
  models: Model[];
  benchmarks: BenchmarkScore[];
  sotaScores: SotaScore[];
  gpuSource: GpuSource;
}

export interface BenchmarkType {
  name: string;
  displayName: string;
}

export interface BenchmarkGroup {
  group: string;
  groupDisplay: string;
  types: BenchmarkType[];
}

interface SnapshotIndex {
  snapshots: string[];
  latest: string | null;
}

/** Fetch all JSON data files from public/data/. */
export function useData(): { data: AppData | null; loading: boolean } {
  const [data, setData] = useState<AppData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        // Load GPUs, models, source metadata, and snapshot index in parallel
        const [gpus, models, gpuSource, index] = await Promise.all([
          fetch(`${BASE_PATH}/data/gpus.json`).then((r) => r.json()),
          fetch(`${BASE_PATH}/data/models.json`).then((r) => r.json()),
          fetch(`${BASE_PATH}/data/gpu_source.json`)
            .then((r) => r.json() as Promise<GpuSource>)
            .catch(() => DEFAULT_GPU_SOURCE),
          fetch(`${BASE_PATH}/data/snapshots/index.json`)
            .then((r) => r.json() as Promise<SnapshotIndex>)
            .catch(() => null),
        ]);

        let benchmarks: BenchmarkScore[] = [];
        let sotaScores: SotaScore[] = [];

        if (index?.latest) {
          // Load from latest snapshot
          const base = `${BASE_PATH}/data/snapshots/${index.latest}`;
          [benchmarks, sotaScores] = await Promise.all([
            fetch(`${base}/benchmarks.json`).then((r) => r.json()),
            fetch(`${base}/sota_scores.json`)
              .then((r) => r.json())
              .catch(() => []),
          ]);
        }

        setData({ gpus, models, benchmarks, sotaScores, gpuSource });
      } catch {
        setData({ gpus: [], models: [], benchmarks: [], sotaScores: [], gpuSource: DEFAULT_GPU_SOURCE });
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return { data, loading };
}

/** Get unique GPU type names sorted alphabetically. */
export function getGpuTypes(gpus: GpuOffering[]): string[] {
  return [...new Set(gpus.map((g) => g.gpu_name))].sort();
}

/** Get unique location names sorted by number of offerings (most first). */
export function getLocations(gpus: GpuOffering[]): string[] {
  const counts = new Map<string, number>();
  for (const g of gpus) {
    counts.set(g.location, (counts.get(g.location) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([loc]) => loc);
}

/** Get all offerings for a specific GPU type. */
export function getOfferings(
  gpus: GpuOffering[],
  gpuType: string,
): GpuOffering[] {
  return gpus.filter((g) => g.gpu_name === gpuType);
}

/** Build benchmark groups from benchmark data. */
export function getBenchmarkGroups(benchmarks: BenchmarkScore[]): BenchmarkGroup[] {
  const groupMap = new Map<string, { groupDisplay: string; types: Map<string, string> }>();

  for (const b of benchmarks) {
    const key = b.benchmark_group;
    if (!groupMap.has(key)) {
      groupMap.set(key, {
        groupDisplay: b.benchmark_group_display,
        types: new Map(),
      });
    }
    const entry = groupMap.get(key)!;
    if (!entry.types.has(b.benchmark_name)) {
      entry.types.set(b.benchmark_name, b.benchmark_display_name);
    }
  }

  return Array.from(groupMap.entries()).map(([group, { groupDisplay, types }]) => ({
    group,
    groupDisplay,
    types: Array.from(types.entries()).map(([name, displayName]) => ({
      name,
      displayName,
    })),
  }));
}

/** Keep only the cheapest offering per (gpu_name, vram_gb, gpu_count). */
export function deduplicateGpus(gpus: GpuOffering[]): GpuOffering[] {
  const best = new Map<string, GpuOffering>();
  for (const g of gpus) {
    const key = `${g.gpu_name}|${g.vram_gb}|${g.gpu_count}`;
    const existing = best.get(key);
    if (!existing || g.price_per_hour < existing.price_per_hour) {
      best.set(key, g);
    }
  }
  return [...best.values()];
}

/** Filter benchmarks by a specific benchmark type name. */
export function filterBenchmarksByType(
  benchmarks: BenchmarkScore[],
  benchmarkName: string,
): BenchmarkScore[] {
  return benchmarks.filter((b) => b.benchmark_name === benchmarkName);
}
