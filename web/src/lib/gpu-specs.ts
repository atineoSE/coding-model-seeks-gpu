/**
 * GPU Throughput Specifications
 *
 * Reference data for GPU compute and memory bandwidth capabilities.
 * Used for calculating decode throughput and team capacity estimates.
 *
 * Sources:
 * - NVIDIA datasheets
 * - llm-capacity skill reference specs
 * - TechPowerUp GPU database
 */

export interface GpuThroughputSpec {
  /** FP16 TFLOPS (Tensor Core performance) */
  fp16_tflops: number;

  /** Memory bandwidth in TB/s */
  memory_bandwidth_tb_s: number;

  /** NVLink bandwidth in GB/s (null for PCIe-only GPUs) */
  nvlink_bandwidth_gb_s: number | null;

  /** FP8 support multiplier (2x for Hopper and newer, 1x for older) */
  fp8_multiplier: number;

  /** Architecture generation (for reference) */
  architecture: string;
}

/**
 * GPU throughput specifications by GPU name.
 *
 * Note: Decode throughput is typically memory-bandwidth limited, not compute limited.
 * The memory_bandwidth_tb_s is the most critical spec for LLM inference.
 */
export const GPU_THROUGHPUT_SPECS: Record<string, GpuThroughputSpec> = {
  // Hopper generation (H-series) - Latest, most capable
  "H100": {
    fp16_tflops: 989,
    memory_bandwidth_tb_s: 3.35,
    nvlink_bandwidth_gb_s: 900,
    fp8_multiplier: 2,
    architecture: "Hopper",
  },
  "H200": {
    fp16_tflops: 989,
    memory_bandwidth_tb_s: 4.8,  // Major upgrade over H100
    nvlink_bandwidth_gb_s: 900,
    fp8_multiplier: 2,
    architecture: "Hopper",
  },
  "GH200": {
    fp16_tflops: 989,
    memory_bandwidth_tb_s: 4.0,  // Grace Hopper Superchip
    nvlink_bandwidth_gb_s: 900,
    fp8_multiplier: 2,
    architecture: "Hopper",
  },

  // Blackwell generation (B-series) - Newest
  "B200": {
    fp16_tflops: 1800,  // Estimated, Blackwell is ~2x Hopper
    memory_bandwidth_tb_s: 8.0,  // Estimated HBM3e bandwidth
    nvlink_bandwidth_gb_s: 1800,  // NVLink 5.0
    fp8_multiplier: 2,
    architecture: "Blackwell",
  },
  "B300": {
    fp16_tflops: 2000,  // Estimated
    memory_bandwidth_tb_s: 8.0,  // Estimated
    nvlink_bandwidth_gb_s: 1800,
    fp8_multiplier: 2,
    architecture: "Blackwell",
  },

  // Ampere generation (A-series)
  "A100": {
    fp16_tflops: 312,
    memory_bandwidth_tb_s: 1.555,  // 40GB variant
    nvlink_bandwidth_gb_s: 600,
    fp8_multiplier: 1,
    architecture: "Ampere",
  },
  "A100_80G": {
    fp16_tflops: 312,
    memory_bandwidth_tb_s: 2.0,  // 80GB variant
    nvlink_bandwidth_gb_s: 600,
    fp8_multiplier: 1,
    architecture: "Ampere",
  },
  "A10": {
    fp16_tflops: 125,
    memory_bandwidth_tb_s: 0.6,
    nvlink_bandwidth_gb_s: null,  // PCIe only
    fp8_multiplier: 1,
    architecture: "Ampere",
  },
  "A16": {
    fp16_tflops: 71,  // Lower performance, multi-instance GPU
    memory_bandwidth_tb_s: 0.2,
    nvlink_bandwidth_gb_s: null,
    fp8_multiplier: 1,
    architecture: "Ampere",
  },
  "A4000": {
    fp16_tflops: 153,
    memory_bandwidth_tb_s: 0.448,
    nvlink_bandwidth_gb_s: null,
    fp8_multiplier: 1,
    architecture: "Ampere",
  },
  "A5000": {
    fp16_tflops: 222,
    memory_bandwidth_tb_s: 0.768,
    nvlink_bandwidth_gb_s: 112.5,  // NVLink available
    fp8_multiplier: 1,
    architecture: "Ampere",
  },
  "A6000": {
    fp16_tflops: 309,
    memory_bandwidth_tb_s: 0.768,
    nvlink_bandwidth_gb_s: 112.5,
    fp8_multiplier: 1,
    architecture: "Ampere",
  },

  // Ada Lovelace generation (L-series and RTX 40/60 series)
  "L4": {
    fp16_tflops: 121,
    memory_bandwidth_tb_s: 0.3,
    nvlink_bandwidth_gb_s: null,
    fp8_multiplier: 1,
    architecture: "Ada Lovelace",
  },
  "L40": {
    fp16_tflops: 362,
    memory_bandwidth_tb_s: 0.864,
    nvlink_bandwidth_gb_s: null,
    fp8_multiplier: 1,
    architecture: "Ada Lovelace",
  },
  "L40S": {
    fp16_tflops: 362,
    memory_bandwidth_tb_s: 0.864,
    nvlink_bandwidth_gb_s: null,  // PCIe only
    fp8_multiplier: 1,
    architecture: "Ada Lovelace",
  },
  "RTX4090": {
    fp16_tflops: 330,
    memory_bandwidth_tb_s: 1.008,
    nvlink_bandwidth_gb_s: null,
    fp8_multiplier: 1,
    architecture: "Ada Lovelace",
  },
  "RTX5090": {
    fp16_tflops: 450,  // Estimated, not yet released
    memory_bandwidth_tb_s: 1.4,  // Estimated GDDR7
    nvlink_bandwidth_gb_s: null,
    fp8_multiplier: 1,
    architecture: "Ada Lovelace",
  },
  "RTX6000Ada": {
    fp16_tflops: 411,
    memory_bandwidth_tb_s: 0.960,
    nvlink_bandwidth_gb_s: 112.5,
    fp8_multiplier: 1,
    architecture: "Ada Lovelace",
  },
  "RTXPro6000": {
    fp16_tflops: 411,  // Same as RTX6000Ada
    memory_bandwidth_tb_s: 0.960,
    nvlink_bandwidth_gb_s: 112.5,
    fp8_multiplier: 1,
    architecture: "Ada Lovelace",
  },

  // Volta generation (V-series) - Older
  "V100": {
    fp16_tflops: 125,
    memory_bandwidth_tb_s: 0.9,
    nvlink_bandwidth_gb_s: 300,
    fp8_multiplier: 1,
    architecture: "Volta",
  },
  "V100_32G": {
    fp16_tflops: 125,
    memory_bandwidth_tb_s: 0.9,
    nvlink_bandwidth_gb_s: 300,
    fp8_multiplier: 1,
    architecture: "Volta",
  },
};

/**
 * Get GPU throughput spec by name.
 * Returns null if GPU not found in specs.
 */
export function getGpuThroughputSpec(gpuName: string): GpuThroughputSpec | null {
  return GPU_THROUGHPUT_SPECS[gpuName] || null;
}

/**
 * Check if GPU supports FP8 compute (Hopper and newer).
 */
export function supportsFp8(gpuName: string): boolean {
  const spec = getGpuThroughputSpec(gpuName);
  return spec?.fp8_multiplier === 2;
}

const FP8_KV_CACHE_ARCHITECTURES = new Set(["Ada Lovelace", "Hopper", "Blackwell"]);

/**
 * Check if GPU supports FP8 KV cache (Ada Lovelace, Hopper, and Blackwell).
 *
 * FP8 KV cache support is broader than FP8 compute — Ada Lovelace GPUs
 * (L4, L40S, RTX 4090, etc.) support FP8 KV cache but not FP8 compute.
 */
export function supportsFp8KvCache(gpuName: string): boolean {
  const spec = getGpuThroughputSpec(gpuName);
  if (!spec) return false;
  return FP8_KV_CACHE_ARCHITECTURES.has(spec.architecture);
}

/**
 * Check if a GPU type supports NVLink interconnect.
 * Derived from GPU throughput specs (nvlink_bandwidth_gb_s).
 */
export function gpuHasNvLink(gpuName: string): boolean {
  const spec = getGpuThroughputSpec(gpuName);
  return spec?.nvlink_bandwidth_gb_s !== null && spec?.nvlink_bandwidth_gb_s !== undefined;
}
