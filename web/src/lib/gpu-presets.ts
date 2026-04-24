import type { GpuOffering, PresetGpuConfig } from "@/types";

export function buildGpuPresets(gpus: GpuOffering[]): PresetGpuConfig[] {
  return gpus
    .filter(g => g.total_vram_gb >= 192 || g.gpu_count >= 4)
    .sort((a, b) => a.price_per_hour - b.price_per_hour)
    .map(g => ({
      label: `${g.gpu_count}× ${g.gpu_name} ${g.vram_gb}GB`,
      gpuName: g.gpu_name,
      gpuCount: g.gpu_count,
      vramPerGpu: g.vram_gb,
      totalVramGb: g.total_vram_gb,
      interconnect: g.interconnect,
    }));
}
