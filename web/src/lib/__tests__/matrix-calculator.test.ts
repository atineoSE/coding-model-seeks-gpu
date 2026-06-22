import { describe, it, expect } from "vitest";
import type { Model, GpuOffering, BenchmarkScore, SotaScore } from "@/types";
import {
  calculatePerformanceMatrix,
  calculateBudgetMatrix,
  calculateBudgetChartData,
  calculateUnrankedMatrix,
  calcGpuSetupStats,
  DEFAULT_ADVANCED_SETTINGS,
} from "../matrix-calculator";
import { WEIGHT_OVERHEAD_FACTOR, gpusNeeded, getModelMemory } from "../calculations";
import { CONCURRENCY_TIERS } from "../concurrency-tiers";

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const GLM_47: Model = {
  model_name: "GLM-4.7",
  learnable_params_b: 352.8,
  active_params_b: 33.7,
  architecture: "MoE",
  context_length: 202752,
  precision: "BF16",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 92,
  num_kv_layers: null,
  num_kv_heads: 8,
  head_dim: 128,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
  model_url: null,
};

const MINIMAX_M25: Model = {
  model_name: "MiniMax-M2.5",
  learnable_params_b: 228.7,
  active_params_b: 11.1,
  architecture: "MoE",
  context_length: 196608,
  precision: "FP8",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 62,
  num_kv_layers: null,
  num_kv_heads: 8,
  head_dim: 128,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
  model_url: null,
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

describe("performance persona scaled fallback", () => {
  // x1 H100 offering — required for findScaledGpuSetups (only considers x1 offerings)
  const H100_X1: GpuOffering = {
    gpu_name: "H100",
    vram_gb: 80,
    gpu_count: 1,
    total_vram_gb: 80,
    price_per_hour: 2.5,
    currency: "USD",
    provider: "test",
    instance_name: "test-x1",
    location: "test-region",
    interconnect: null,
  };

  const allGpus = [H100_X1];
  const allModels = [MINIMAX_M25];
  const benchmarks = [BENCHMARK_MINIMAX];
  const sotaScores = [SOTA];

  it("scales up from x1 offering and marks setups as projected", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(1);
    // At least one cell should use the scaled fallback (no real offering can serve high concurrency)
    const scaledCells = matrix[0].filter(
      (cell) => cell.gpuSetups.length > 0 && cell.gpuSetups[0].isProjected,
    );
    expect(scaledCells.length).toBeGreaterThan(0);
  });

  it("projected setups never exceed 8 GPUs", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    for (const cell of matrix[0]) {
      for (const setup of cell.gpuSetups) {
        if (setup.isProjected) {
          expect(setup.gpuCount).toBeLessThanOrEqual(8);
        }
      }
    }
  });

  it("shows exceedsCapacity when 8 GPUs is not enough", () => {
    // GLM-4.7 is a huge model (352.8B BF16 → ~706 GB weights).
    // With 80 GB/GPU, it needs ceil(706*1.15/80) = 11 GPUs minimum.
    // Since the cap is 8, it can't fit at all → exceedsCapacity.
    const matrix = calculatePerformanceMatrix(
      allGpus, [GLM_47], [BENCHMARK_GLM], sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(1);
    for (const cell of matrix[0]) {
      expect(cell.exceedsCapacity).toBe(true);
    }
  });

  it("cells with projected setups have throughput and utilization populated", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    for (const cell of matrix[0]) {
      if (cell.gpuSetups.length > 0) {
        expect(cell.decodeThroughputTokS).not.toBeNull();
        expect(cell.decodeThroughputTokS).toBeGreaterThan(0);
        expect(cell.utilization).not.toBeNull();
        expect(cell.utilization).toBeGreaterThan(0);
        expect(cell.utilization).toBeLessThanOrEqual(1.0);
      }
    }
  });

  it("ignores GPU types without a x1 offering", () => {
    // Only a 4×H100 offering, no x1 → findScaledGpuSetups returns nothing
    const MULTI_ONLY: GpuOffering = {
      gpu_name: "H100",
      vram_gb: 80,
      gpu_count: 4,
      total_vram_gb: 320,
      price_per_hour: 10,
      currency: "USD",
      provider: "test",
      instance_name: "test-multi",
      location: "test-region",
      interconnect: null,
    };
    const matrix = calculatePerformanceMatrix(
      [MULTI_ONLY], allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(1);
    // All tiers should show exceedsCapacity since the 4×H100 offering
    // may not directly serve concurrency, and no x1 exists for fallback
    for (const cell of matrix[0]) {
      if (cell.gpuSetups.length === 0) {
        expect(cell.exceedsCapacity).toBe(true);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// calculateBudgetChartData — sizing-first, includes unranked models
// ---------------------------------------------------------------------------

describe("calculateBudgetChartData (unranked models)", () => {
  // A small, sized model with NO benchmark entry → unranked. 14B dense BF16:
  // 28 GB weights × 1.15 ≈ 33 GB → fits a single H100.
  const SMALL_UNRANKED: Model = {
    model_name: "Tiny-14B",
    learnable_params_b: 14,
    active_params_b: null,
    architecture: "Dense",
    context_length: 131072,
    precision: "BF16",
    routed_expert_params_b: null,
    attention_type: "GQA",
    num_hidden_layers: 40,
    num_kv_layers: null,
    num_kv_heads: 8,
    head_dim: 128,
    kv_lora_rank: null,
    qk_rope_head_dim: null,
    hf_model_id: null,
    model_url: null,
  };

  const gpuConfig = {
    label: "12×H100",
    gpuName: "H100",
    gpuCount: 12,
    vramPerGpu: 80,
    totalVramGb: 960,
    interconnect: "nvlink" as const,
  };
  // GLM (72.3) and MiniMax (65.0) are ranked; Tiny-14B has no benchmark.
  const allModels = [GLM_47, MINIMAX_M25, SMALL_UNRANKED];
  const benchmarks = [BENCHMARK_GLM, BENCHMARK_MINIMAX];
  const sotaScores = [SOTA];

  function build() {
    return calculateBudgetChartData(
      gpuConfig,
      allModels,
      benchmarks,
      sotaScores,
      "swe_bench_verified",
      0.9,
      DEFAULT_ADVANCED_SETTINGS,
    );
  }

  it("includes a fitting model that has no benchmark and flags it unranked", () => {
    const data = build();
    const tiny = data.find((d) => d.modelName === "Tiny-14B");
    expect(tiny).toBeDefined();
    expect(tiny!.isUnranked).toBe(true);
    expect(tiny!.fits).toBe(true);
  });

  it("never emits a 0 score for an unranked model (null, not 0)", () => {
    const data = build();
    const tiny = data.find((d) => d.modelName === "Tiny-14B")!;
    expect(tiny.benchmarkScore).toBeNull();
    expect(tiny.percentOfSota).toBeNull();
    // Explicitly: the missing score must not be coerced to 0.
    expect(tiny.benchmarkScore).not.toBe(0);
    expect(tiny.percentOfSota).not.toBe(0);
  });

  it("orders unranked models after ranked ones", () => {
    const data = build();
    const tinyIdx = data.findIndex((d) => d.modelName === "Tiny-14B");
    const glmIdx = data.findIndex((d) => d.modelName === "GLM-4.7");
    const miniIdx = data.findIndex((d) => d.modelName === "MiniMax-M2.5");
    expect(glmIdx).toBeGreaterThanOrEqual(0);
    expect(miniIdx).toBeGreaterThanOrEqual(0);
    expect(tinyIdx).toBeGreaterThan(glmIdx);
    expect(tinyIdx).toBeGreaterThan(miniIdx);
    // Ranked models keep score-descending order (GLM 72.3 > MiniMax 65.0).
    expect(glmIdx).toBeLessThan(miniIdx);
  });

  it("keeps the ≥50%-of-SOTA filter for ranked models but not unranked", () => {
    // A ranked model well below 50% of SOTA (80) should be dropped...
    const lowRanked: BenchmarkScore = {
      ...BENCHMARK_MINIMAX,
      model_name: "MiniMax-M2.5",
      score: 20.0, // 25% of SOTA → excluded
    };
    const data = calculateBudgetChartData(
      gpuConfig,
      allModels,
      [BENCHMARK_GLM, lowRanked],
      sotaScores,
      "swe_bench_verified",
      0.9,
      DEFAULT_ADVANCED_SETTINGS,
    );
    expect(data.some((d) => d.modelName === "MiniMax-M2.5")).toBe(false);
    // ...while the unranked model is still surfaced.
    expect(data.some((d) => d.modelName === "Tiny-14B")).toBe(true);
  });

  it("computes a real % of SOTA for ranked models", () => {
    const data = build();
    const glm = data.find((d) => d.modelName === "GLM-4.7")!;
    expect(glm.isUnranked).toBe(false);
    expect(glm.benchmarkScore).toBe(72.3);
    // 72.3 / 80 = 90.375%
    expect(glm.percentOfSota).toBeCloseTo(90.375, 3);
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

// ---------------------------------------------------------------------------
// calculateUnrankedMatrix
// ---------------------------------------------------------------------------

describe("calculateUnrankedMatrix", () => {
  // No benchmark scores in the "frontend" category → both models are unranked there.
  const models = [GLM_47, MINIMAX_M25];
  const gpus = [H100_OFFERING];

  it("returns one matrix row per sized, unranked model, one cell per tier", () => {
    const rows = calculateUnrankedMatrix(gpus, models, [], "frontend");
    expect(rows.map((row) => row[0].model.model_name).sort()).toEqual([
      "GLM-4.7",
      "MiniMax-M2.5",
    ]);
    for (const row of rows) {
      expect(row).toHaveLength(CONCURRENCY_TIERS.length);
    }
  });

  it("marks cells unranked with no score fields (never faked)", () => {
    const rows = calculateUnrankedMatrix(gpus, models, [], "frontend");
    for (const row of rows) {
      for (const cell of row) {
        expect(cell.isUnranked).toBe(true);
        expect(cell.benchmark).toBeNull();
        expect(cell.percentOfSota).toBeNull();
        expect(cell.sotaScore).toBeNull();
        expect(cell.totalBenchmarkCost).toBeNull();
      }
    }
  });

  it("excludes models that are ranked in the category", () => {
    const benchmarks: BenchmarkScore[] = [
      { ...BENCHMARK_GLM, benchmark_name: "frontend" },
    ];
    const rows = calculateUnrankedMatrix(gpus, models, benchmarks, "frontend");
    // GLM-4.7 now has a frontend score → only MiniMax-M2.5 remains unranked.
    expect(rows.map((row) => row[0].model.model_name)).toEqual(["MiniMax-M2.5"]);
  });

  it("sorts by minimum VRAM descending (biggest first)", () => {
    const rows = calculateUnrankedMatrix(gpus, models, [], "frontend");
    // GLM-4.7 (352.8b @ BF16) needs more VRAM than MiniMax-M2.5 (228.7b @ FP8).
    expect(rows[0][0].model.model_name).toBe("GLM-4.7");
    expect(rows[1][0].model.model_name).toBe("MiniMax-M2.5");
  });
});

// ---------------------------------------------------------------------------
// calcGpuSetupStats — KV cache dtype setting
// ---------------------------------------------------------------------------

describe("calcGpuSetupStats — kvCachePrecision", () => {
  it("defaults to 2 bytes (fp16) → matches vLLM kv_cache_dtype=auto bf16 reality", () => {
    expect(DEFAULT_ADVANCED_SETTINGS.kvCachePrecision).toBe("fp16");
  });

  it("opting into 1-byte FP8 KV roughly doubles streams on a Hopper+ GPU", () => {
    const args = [MINIMAX_M25, "H100", 12, 960, null] as const;
    const fp16 = calcGpuSetupStats(...args, {
      ...DEFAULT_ADVANCED_SETTINGS,
      kvCachePrecision: "fp16",
    });
    const fp8 = calcGpuSetupStats(...args, {
      ...DEFAULT_ADVANCED_SETTINGS,
      kvCachePrecision: "auto",
    });
    // Only the KV denominator changes, so FP8 (1 B) ≈ 2× the fp16 (2 B) streams.
    expect(fp16.maxConcurrentStreams).toBeGreaterThan(0);
    expect(fp8.maxConcurrentStreams).toBeGreaterThan(fp16.maxConcurrentStreams);
    // Decode is bandwidth-bound and KV-dtype-independent.
    expect(fp8.decodeThroughputTokS).toBe(fp16.decodeThroughputTokS);
  });
});
