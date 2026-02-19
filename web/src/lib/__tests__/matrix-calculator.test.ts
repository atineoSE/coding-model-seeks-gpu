import { describe, it, expect } from "vitest";
import type { Model, GpuOffering, BenchmarkScore, SotaScore } from "@/types";
import {
  calculatePerformanceMatrix,
  calculateBudgetMatrix,
} from "../matrix-calculator";
import { WEIGHT_OVERHEAD_FACTOR, gpusNeeded, getModelMemory, resolveModelPrecision } from "../calculations";
import { CONCURRENCY_TIERS } from "../concurrency-tiers";

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const GLM_47: Model = {
  model_name: "GLM-4.7",
  published_param_count_b: null,
  learnable_params_b: 352.8,
  active_params_b: 33.7,
  architecture: "MoE",
  context_length: 202752,
  precision: "BF16",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 92,
  num_kv_heads: 8,
  head_dim: 128,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
};

const MINIMAX_M25: Model = {
  model_name: "MiniMax-M2.5",
  published_param_count_b: 228.7,
  learnable_params_b: 228.7,
  active_params_b: 11.1,
  architecture: "MoE",
  context_length: 196608,
  precision: "FP8",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 62,
  num_kv_heads: 8,
  head_dim: 128,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
};

const BENCHMARK_GLM: BenchmarkScore = {
  model_name: "GLM-4.7",
  benchmark_name: "swe_bench_verified",
  benchmark_display_name: "SWE-bench Verified",
  score: 72.3,
  rank: 1,
  cost_per_task: null,
  benchmark_group: "coding",
  benchmark_group_display: "Coding",
};

const BENCHMARK_MINIMAX: BenchmarkScore = {
  model_name: "MiniMax-M2.5",
  benchmark_name: "swe_bench_verified",
  benchmark_display_name: "SWE-bench Verified",
  score: 65.0,
  rank: 2,
  cost_per_task: null,
  benchmark_group: "coding",
  benchmark_group_display: "Coding",
};

const SOTA: SotaScore = {
  benchmark_name: "swe_bench_verified",
  benchmark_display_name: "SWE-bench Verified",
  sota_model_name: "GPT-5",
  sota_score: 80.0,
};

// GPU offering that can fit both models at their native precision with overhead
const H100_OFFERING: GpuOffering = {
  gpu_name: "H100",
  vram_gb: 80,
  gpu_count: 12,
  total_vram_gb: 960,
  price_per_hour: 30,
  currency: "USD",
  provider: "test",
  instance_name: "test-instance",
  location: "test-region",
  interconnect: null,
};

// ---------------------------------------------------------------------------
// Overhead is applied when checking model fit
// ---------------------------------------------------------------------------

describe("model fit with overhead", () => {
  it("gpusNeeded uses WEIGHT_OVERHEAD_FACTOR", () => {
    // MiniMax-M2.5 at fp16: 228.7 × 2 = 457.4 GB
    // Without overhead: ceil(457.4 / 80) = 6 GPUs
    // With overhead: ceil(457.4 × 1.15 / 80) = ceil(526.01 / 80) = 7 GPUs
    const memGb = getModelMemory(MINIMAX_M25, "fp16")!;
    expect(gpusNeeded(memGb, 80)).toBe(6);
    expect(gpusNeeded(memGb * WEIGHT_OVERHEAD_FACTOR, 80)).toBe(7);
  });
});

// ---------------------------------------------------------------------------
// calculatePerformanceMatrix
// ---------------------------------------------------------------------------

describe("calculatePerformanceMatrix", () => {
  const allGpus = [H100_OFFERING];
  const allModels = [GLM_47, MINIMAX_M25];
  const benchmarks = [BENCHMARK_GLM, BENCHMARK_MINIMAX];
  const sotaScores = [SOTA];

  it("returns 4 columns (concurrency tiers)", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(2);
    expect(matrix[0].length).toBe(4);
  });

  it("returns rows sorted by benchmark score descending", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(2);
    // First row should be GLM (score 72.3 > MiniMax 65.0)
    expect(matrix[0][0].model.model_name).toBe("GLM-4.7");
    expect(matrix[1][0].model.model_name).toBe("MiniMax-M2.5");
  });

  it("computes percent of SOTA correctly", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    // GLM: 72.3 / 80.0 = 0.90375
    expect(matrix[0][0].percentOfSota).toBeCloseTo(0.90375, 4);
  });

  it("last column is agent_swarm with midpoint 150", () => {
    const lastTier = CONCURRENCY_TIERS[3];
    expect(lastTier.key).toBe("agent_swarm");
    expect(lastTier.midpoint).toBe(150);
  });

  it("uses costPerStreamPerMonth and maxConcurrentStreams fields", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    // Check that the first non-empty cell has the new field names
    const cell = matrix[0][0];
    if (cell.gpuSetups.length > 0) {
      expect(cell.gpuSetups[0]).toHaveProperty("costPerStreamPerMonth");
      expect(cell.gpuSetups[0]).toHaveProperty("maxConcurrentStreams");
    }
  });
});

// ---------------------------------------------------------------------------
// calculateBudgetMatrix
// ---------------------------------------------------------------------------

describe("calculateBudgetMatrix", () => {
  const gpuConfig = {
    label: "12×H100",
    gpuName: "H100",
    gpuCount: 12,
    vramPerGpu: 80,
    totalVramGb: 960,
    interconnect: "nvlink" as const,  // kept for PresetGpuConfig type
  };
  const allGpus = [H100_OFFERING];
  const allModels = [GLM_47, MINIMAX_M25];
  const benchmarks = [BENCHMARK_GLM, BENCHMARK_MINIMAX];
  const sotaScores = [SOTA];

  it("excludes models that don't fit with overhead", () => {
    // Tiny config: 2×H100 = 160 GB
    // GLM-4.7 at BF16 + overhead: 352.8×2×1.15 = 811 GB — doesn't fit
    // MiniMax at FP8 + overhead: 228.7×1×1.15 = 263 GB — doesn't fit
    const tinyConfig = { ...gpuConfig, gpuCount: 2, vramPerGpu: 80, totalVramGb: 160 };
    const matrix = calculateBudgetMatrix(
      tinyConfig, allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(0);
  });

  it("includes models that fit with overhead", () => {
    const matrix = calculateBudgetMatrix(
      gpuConfig, allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(2);
  });

  it("returns 4 columns per row", () => {
    const matrix = calculateBudgetMatrix(
      gpuConfig, allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    for (const row of matrix) {
      expect(row.length).toBe(4);
    }
  });

  it("costPerStreamPerMonth = monthlyCost / midpoint when not exceeding capacity", () => {
    const matrix = calculateBudgetMatrix(
      gpuConfig, allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    const monthlyCost = H100_OFFERING.price_per_hour * 720;
    for (const row of matrix) {
      for (let i = 0; i < row.length; i++) {
        const cell = row[i];
        if (!cell.exceedsCapacity && cell.costPerStreamPerMonth !== null) {
          const tier = CONCURRENCY_TIERS[i];
          expect(cell.costPerStreamPerMonth).toBeCloseTo(monthlyCost / tier.midpoint, 0);
        }
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Performance persona: never exceeds capacity (scaled fallback)
// ---------------------------------------------------------------------------

describe("performance persona never shows exceedsCapacity", () => {
  // Use a small 4×H100 offering that can't directly serve 50 streams
  const SMALL_H100: GpuOffering = {
    gpu_name: "H100",
    vram_gb: 80,
    gpu_count: 4,
    total_vram_gb: 320,
    price_per_hour: 10,
    currency: "USD",
    provider: "test",
    instance_name: "test-instance",
    location: "test-region",
    interconnect: null,
  };

  const allGpus = [SMALL_H100];
  const allModels = [MINIMAX_M25];
  const benchmarks = [BENCHMARK_MINIMAX];
  const sotaScores = [SOTA];

  it("all cells have exceedsCapacity: false", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(1);
    for (const cell of matrix[0]) {
      expect(cell.exceedsCapacity).toBe(false);
    }
  });

  it("scaled cells have gpuCount >= available offering", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    // The "agent_swarm" tier (midpoint=50) likely requires more than 4 GPUs
    const swarmCell = matrix[0][3]; // last column
    expect(swarmCell.gpuSetups.length).toBeGreaterThan(0);
    const setup = swarmCell.gpuSetups[0];
    expect(setup.gpuCount).toBeGreaterThanOrEqual(4);
  });

  it("cells have throughput and utilization populated", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    for (const cell of matrix[0]) {
      expect(cell.decodeThroughputTokS).not.toBeNull();
      expect(cell.decodeThroughputTokS).toBeGreaterThan(0);
      expect(cell.utilization).not.toBeNull();
      expect(cell.utilization).toBeGreaterThan(0);
      expect(cell.utilization).toBeLessThanOrEqual(1.0);
    }
  });
});

// ---------------------------------------------------------------------------
// Per-stream throughput with pipeline parallelism
// ---------------------------------------------------------------------------

describe("per-stream throughput with PP > 1", () => {
  // 12 H100s → PP=2, so pipeline bubble fill improves with more concurrency
  const allGpus = [H100_OFFERING];
  const allModels = [MINIMAX_M25];
  const benchmarks = [BENCHMARK_MINIMAX];
  const sotaScores = [SOTA];

  it("throughput per stream is non-decreasing across tiers (better PP fill)", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(1);
    const row = matrix[0];

    // Collect non-null throughput values
    const throughputs = row
      .map((cell) => cell.decodeThroughputTokS)
      .filter((t): t is number => t !== null);

    // With PP=2, per-stream throughput improves as concurrency increases
    // (pipeline bubble is amortized over more micro-batches)
    for (let i = 1; i < throughputs.length; i++) {
      expect(throughputs[i]).toBeGreaterThanOrEqual(throughputs[i - 1]);
    }
  });
});
