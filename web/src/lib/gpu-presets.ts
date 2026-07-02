import type { GpuOffering, PresetGpuConfig, Model, BenchmarkScore } from "@/types";
import { getModelMemory, resolveModelPrecision, gpusNeeded, WEIGHT_OVERHEAD_FACTOR } from "./calculations";
import { getGpuVram } from "./gpu-specs";

const MAX_PRESETS = 8;

function getTopOpenModels(models: Model[], benchmarks: BenchmarkScore[], count: number): Model[] {
  const openModelNames = new Set(models.map(m => m.model_name));

  const overallBenchmarks = benchmarks.filter(
    b => b.benchmark_name === "overall" && b.score !== null && openModelNames.has(b.model_name)
  );

  return overallBenchmarks
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
    .slice(0, count)
    .map(b => models.find(m => m.model_name === b.model_name))
    .filter((m): m is Model => m !== undefined);
}

function fitsOnGpu(model: Model, gpu: GpuOffering): boolean {
  const precision = resolveModelPrecision(model);
  const memGb = getModelMemory(model, precision);
  return memGb !== null && gpusNeeded(memGb * WEIGHT_OVERHEAD_FACTOR, gpu.vram_gb) <= gpu.gpu_count;
}

export function buildGpuPresets(
  gpus: GpuOffering[],
  models: Model[],
  benchmarks: BenchmarkScore[],
): PresetGpuConfig[] {
  const topModels = getTopOpenModels(models, benchmarks, 3);

  const ALLOWED_GPU_COUNTS = new Set([1, 4, 8, 16]);

  const filtered = topModels.length > 0
    ? gpus.filter(g => ALLOWED_GPU_COUNTS.has(g.gpu_count) && topModels.some(m => fitsOnGpu(m, g)))
    : gpus.filter(g => ALLOWED_GPU_COUNTS.has(g.gpu_count));

  const seenConfig = new Set<string>();
  return filtered
    .sort((a, b) => a.price_per_hour - b.price_per_hour)
    // Collapse offerings that map to the same config (same GPU × count). After
    // datasheet-VRAM normalization their labels are identical, which would give
    // duplicate React keys; keep the cheapest (first after the sort).
    .filter(g => {
      const key = `${g.gpu_name}|${g.gpu_count}`;
      if (seenConfig.has(key)) return false;
      seenConfig.add(key);
      return true;
    })
    .slice(0, MAX_PRESETS)
    .map(g => {
      // Use the GPU's datasheet VRAM, not the offering's reported value, which
      // can carry float noise (e.g. a B200 at 179.0615…GB instead of 180).
      const vram = getGpuVram(g.gpu_name) ?? g.vram_gb;
      return {
        label: `${g.gpu_count}× ${g.gpu_name} ${vram}GB`,
        gpuName: g.gpu_name,
        gpuCount: g.gpu_count,
        vramPerGpu: vram,
        totalVramGb: vram * g.gpu_count,
        interconnect: g.interconnect,
      };
    });
}
