import type { PresetGpuConfig } from "@/types";

/**
 * Pre-selected GPU configurations for the Budget persona.
 *
 * Every preset can serve at least one of the top-3 open-source models
 * (MiniMax-M2.5, Kimi-K2.5, DeepSeek-V3.2-Reasoner) up to the Agent Swarm
 * concurrency tier (50 streams).
 *
 * Two configs per GPU type: a smaller entry-point and a larger one.
 */
export const GPU_PRESETS: PresetGpuConfig[] = [
  // RTX PRO — workstation Blackwell, NVLink-capable
  { label: "8× RTX PRO 6000 96GB", gpuName: "RTXPRO6000", gpuCount: 8, vramPerGpu: 96, totalVramGb: 768, interconnect: "nvlink" },

  // Blackwell (B-series) — newest, highest bandwidth
  { label: "2× B300 192GB", gpuName: "B300", gpuCount: 2, vramPerGpu: 192, totalVramGb: 384, interconnect: "nvlink" },
  { label: "4× B300 192GB", gpuName: "B300", gpuCount: 4, vramPerGpu: 192, totalVramGb: 768, interconnect: "nvlink" },
  { label: "2× B200 192GB", gpuName: "B200", gpuCount: 2, vramPerGpu: 192, totalVramGb: 384, interconnect: "nvlink" },
  { label: "4× B200 192GB", gpuName: "B200", gpuCount: 4, vramPerGpu: 192, totalVramGb: 768, interconnect: "nvlink" },

  // Hopper (H-series)
  { label: "4× H200 141GB", gpuName: "H200", gpuCount: 4, vramPerGpu: 141, totalVramGb: 564, interconnect: "nvlink" },
  { label: "8× H200 141GB", gpuName: "H200", gpuCount: 8, vramPerGpu: 141, totalVramGb: 1128, interconnect: "nvlink" },
  { label: "8× H100 80GB", gpuName: "H100", gpuCount: 8, vramPerGpu: 80, totalVramGb: 640, interconnect: "nvlink" },
  { label: "16× H100 80GB", gpuName: "H100", gpuCount: 16, vramPerGpu: 80, totalVramGb: 1280, interconnect: "nvlink" },

  // Ampere (A-series) — older, more affordable
  { label: "8× A100 80GB", gpuName: "A100_80G", gpuCount: 8, vramPerGpu: 80, totalVramGb: 640, interconnect: "nvlink" },
  { label: "16× A100 80GB", gpuName: "A100_80G", gpuCount: 16, vramPerGpu: 80, totalVramGb: 1280, interconnect: "nvlink" },
];
