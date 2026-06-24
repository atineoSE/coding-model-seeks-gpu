import { describe, it, expect } from "vitest";
import type { Model, GpuOffering, BenchmarkScore } from "@/types";
import { buildGpuPresets } from "../gpu-presets";

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

// A large top model (~1494 GB of weights) that only fits on big-VRAM pods.
const BIG_MODEL: Model = {
  model_name: "GLM-5.1",
  learnable_params_b: 747,
  active_params_b: 33.7,
  architecture: "MoE",
  context_length: 202752,
  precision: "BF16",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 92,
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: 8,
  head_dim: 128,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  kv_elems_per_token: null,
  hf_model_id: null,
  model_url: null,
  license_name: null,
  license_url: null,
};

const BIG_MODEL_BENCHMARK: BenchmarkScore = {
  model_name: "GLM-5.1",
  benchmark_name: "overall",
  benchmark_display_name: "Overall",
  score: 80,
  rank: 1,
  cost_per_task: null,
  benchmark_group: "coding",
  benchmark_group_display: "Coding",
};

function gpu(overrides: Partial<GpuOffering>): GpuOffering {
  return {
    gpu_name: "H100",
    vram_gb: 80,
    gpu_count: 8,
    total_vram_gb: 640,
    price_per_hour: 20,
    currency: "USD",
    provider: "test",
    instance_name: "test-instance",
    location: "North America",
    interconnect: "NVLink",
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("buildGpuPresets", () => {
  // Regression: the budget view crashed on regions whose catalog has no pod
  // large enough for the top models — buildGpuPresets returned [], the view
  // read [0] (undefined), and dereferencing .vramPerGpu threw.
  it("returns an empty list when no pod can host the top models", () => {
    // 80 GB-class pods (640 GB total) — too small for the ~1719 GB the big
    // model needs after overhead.
    const smallPods: GpuOffering[] = [
      gpu({ gpu_name: "A100-80G", vram_gb: 80, gpu_count: 8, total_vram_gb: 640 }),
      gpu({ gpu_name: "H100", vram_gb: 80, gpu_count: 8, total_vram_gb: 640 }),
    ];

    const presets = buildGpuPresets(smallPods, [BIG_MODEL], [BIG_MODEL_BENCHMARK]);

    expect(presets).toEqual([]);
  });

  it("returns presets when a pod is large enough for a top model", () => {
    const pods: GpuOffering[] = [
      gpu({ gpu_name: "A100-80G", vram_gb: 80, gpu_count: 8, total_vram_gb: 640 }),
      // 96 GB × 16 = 1536 GB on a card big enough for the big model's weights.
      gpu({ gpu_name: "H200", vram_gb: 144, gpu_count: 16, total_vram_gb: 2304 }),
    ];

    const presets = buildGpuPresets(pods, [BIG_MODEL], [BIG_MODEL_BENCHMARK]);

    expect(presets.length).toBeGreaterThan(0);
    expect(presets[0]).toMatchObject({ gpuName: "H200", gpuCount: 16, vramPerGpu: 144 });
  });
});
