import type { GpuOffering, PresetGpuConfig, Model, BenchmarkScore } from "@/types";
import { getModelMemory, resolveModelPrecision, gpusNeeded, WEIGHT_OVERHEAD_FACTOR } from "./calculations";

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

  return filtered
    .sort((a, b) => a.price_per_hour - b.price_per_hour)
    .slice(0, MAX_PRESETS)
    .map(g => ({
      label: `${g.gpu_count}× ${g.gpu_name} ${g.vram_gb}GB`,
      gpuName: g.gpu_name,
      gpuCount: g.gpu_count,
      vramPerGpu: g.vram_gb,
      totalVramGb: g.total_vram_gb,
      interconnect: g.interconnect,
    }));
}
