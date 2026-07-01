import { describe, it, expect } from "vitest";
import type { Model, GpuOffering, BenchmarkScore, SotaScore } from "@/types";
import {
  calculatePerformanceMatrix,
  calculateBudgetMatrix,
  calculateBudgetChartData,
  calculateUnrankedMatrix,
  calcGpuSetupStats,
  calcDeploymentEstimate,
  DEFAULT_ADVANCED_SETTINGS,
  DEFAULT_MEMORY_UTILIZATION,
} from "../matrix-calculator";
import { WEIGHT_OVERHEAD_FACTOR, gpusNeeded, getModelMemory } from "../calculations";
import { getGpuThroughputSpec } from "../gpu-specs";
import { CONCURRENCY_TIERS } from "../concurrency-tiers";
import { PERFORMANCE_COLUMNS } from "../performance-columns";

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
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: 8,
  head_dim: 128,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
  model_url: null,
  kv_elems_per_token: null,
  license_name: null,
  license_url: null,
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
  hidden_size: null,
  num_kv_layers: null,
  num_kv_heads: 8,
  head_dim: 128,
  num_experts: null,
  experts_per_token: null,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
  model_url: null,
  kv_elems_per_token: null,
  license_name: null,
  license_url: null,
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

  it("returns one column per performance operating point (Fit, Scale)", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    expect(matrix.length).toBe(2);
    expect(matrix[0].length).toBe(PERFORMANCE_COLUMNS.length);
    expect(PERFORMANCE_COLUMNS.map((c) => c.key)).toEqual(["fit", "scale"]);
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

  it("the Scale column requires at least 100 operating streams (or exceeds capacity)", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    const scaleIdx = PERFORMANCE_COLUMNS.findIndex((c) => c.key === "scale");
    for (const row of matrix) {
      const cell = row[scaleIdx];
      const setup = cell.gpuSetups[0];
      if (setup) {
        expect(setup.maxConcurrentStreams).toBeGreaterThanOrEqual(100);
      } else {
        expect(cell.exceedsCapacity).toBe(true);
      }
    }
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

  it("cells with projected setups expose a real setup and gated throughput", () => {
    const matrix = calculatePerformanceMatrix(
      allGpus, allModels, benchmarks, sotaScores, "swe_bench_verified",
    );
    for (const cell of matrix[0]) {
      if (cell.gpuSetups.length > 0) {
        // The chosen setup can admit at least one stream at 90% utilization.
        expect(cell.gpuSetups[0].maxConcurrentStreams).toBeGreaterThan(0);
        // Utilization is fixed at 90% and no longer surfaced per cell.
        expect(cell.utilization).toBeNull();
        // Throughput is architecture-gated: a number for modeled archs, null for
        // unsupported (hybrid/sparse) ones — never a fabricated figure.
        if (cell.decodeThroughputTokS !== null) {
          expect(cell.decodeThroughputTokS).toBeGreaterThan(0);
        }
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
    hidden_size: null,
    num_kv_layers: null,
    num_kv_heads: 8,
    head_dim: 128,
    num_experts: null,
    experts_per_token: null,
    kv_lora_rank: null,
    qk_rope_head_dim: null,
    hf_model_id: null,
    model_url: null,
    kv_elems_per_token: null,
    license_name: null,
    license_url: null,
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

  it("returns one matrix row per sized, unranked model, one cell per column", () => {
    const rows = calculateUnrankedMatrix(gpus, models, [], "frontend");
    expect(rows.map((row) => row[0].model.model_name).sort()).toEqual([
      "GLM-4.7",
      "MiniMax-M2.5",
    ]);
    for (const row of rows) {
      expect(row).toHaveLength(PERFORMANCE_COLUMNS.length);
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

// ---------------------------------------------------------------------------
// calcDeploymentEstimate — topology + latency + capacity integration
// ---------------------------------------------------------------------------

describe("calcDeploymentEstimate", () => {
  // A fully-dimensioned MoE model: enough public dims for the latency model
  // (hidden/layers/top-k) and a GQA KV width for the capacity model.
  const MOE: Model = {
    model_name: "Test-MoE-235B",
    learnable_params_b: 235,
    active_params_b: 22,
    architecture: "MoE",
    context_length: 131072,
    precision: "BF16",
    routed_expert_params_b: null,
    attention_type: "GQA",
    num_hidden_layers: 94,
    hidden_size: 4096,
    num_kv_layers: null,
    num_kv_heads: 8,
    head_dim: 128,
    num_experts: 128,
    experts_per_token: 8,
    kv_lora_rank: null,
    qk_rope_head_dim: null,
    hf_model_id: null,
    model_url: null,
    kv_elems_per_token: null,
    license_name: null,
    license_url: null,
  };

  // A dense model of the same dims minus the MoE routing — used to check the
  // `moe` assumption flag flips and the EP all-to-all term drops out.
  const DENSE: Model = {
    ...MOE,
    model_name: "Test-Dense-22B",
    learnable_params_b: 22,
    active_params_b: null,
    architecture: "Dense",
    num_experts: null,
    experts_per_token: null,
  };

  // 8× H100 (nvswitch, 80 GB each) — 640 GB fits the 235B BF16 (~470 GB) MoE.
  function offering(gpuCount: number, gpuName = "H100"): GpuOffering {
    const vram = getGpuThroughputSpec(gpuName)!.memory_size_gb;
    return {
      gpu_name: gpuName,
      vram_gb: vram,
      gpu_count: gpuCount,
      total_vram_gb: vram * gpuCount,
      price_per_hour: 1,
      currency: "USD",
      provider: "test",
      instance_name: "test",
      location: "test",
      interconnect: "nvlink",
    };
  }

  // Smaller context than the 50k default so operating streams are clearly > 0.
  const SETTINGS: typeof DEFAULT_ADVANCED_SETTINGS = {
    ...DEFAULT_ADVANCED_SETTINGS,
    avgInputTokens: 8000,
    avgOutputTokens: 1000,
  };

  it("assembles a full estimate for a fitting MoE layout", () => {
    const est = calcDeploymentEstimate(MOE, offering(8), SETTINGS);
    expect(est).not.toBeNull();
    expect(est!.singleStreamTokS).toBeGreaterThan(0);
    expect(est!.aggregateTokS).toBeGreaterThan(0);
    expect(est!.operatingStreams.high).toBeGreaterThanOrEqual(est!.operatingStreams.low);
    expect(est!.operatingStreams.low).toBeGreaterThan(0);
  });

  it("passes through the interconnect tier, MoE flag, and context assumptions", () => {
    const est = calcDeploymentEstimate(MOE, offering(8), SETTINGS)!;
    expect(est.assumptions.interconnectTier).toBe("nvswitch"); // H100 → nvswitch
    expect(est.assumptions.moe).toBe(true);
    expect(est.assumptions.context).toEqual({
      avgInputTokens: 8000,
      avgOutputTokens: 1000,
    });
  });

  it("aggregate (batched) throughput is the bonus over single-stream", () => {
    const est = calcDeploymentEstimate(MOE, offering(8), SETTINGS)!;
    // High operating batch amortizes the weight read, so the batched roofline
    // clears the single-stream latency-bound rate.
    expect(est.operatingStreams.high).toBeGreaterThanOrEqual(1);
    expect(est.aggregateTokS!).toBeGreaterThan(est.singleStreamTokS!);
  });

  it("caps aggregate at the prefill compute roofline of the used GPUs", () => {
    const est = calcDeploymentEstimate(MOE, offering(8), SETTINGS)!;
    const spec = getGpuThroughputSpec("H100")!;
    // gpusUsed = 8 (nvswitch spans all), 2 FLOPs/param/token forward pass.
    const prefillCeiling = (8 * spec.fp16_tflops * 1e12) / (2 * MOE.active_params_b! * 1e9);
    expect(est.aggregateTokS).toBeLessThanOrEqual(prefillCeiling + 1e-6);
  });

  it("flips the MoE assumption flag for a dense model", () => {
    const est = calcDeploymentEstimate(DENSE, offering(2), SETTINGS)!;
    expect(est).not.toBeNull();
    expect(est.assumptions.moe).toBe(false);
  });

  it("widens the operating-streams band as VRAM (GPU count) grows", () => {
    const small = calcDeploymentEstimate(MOE, offering(6), SETTINGS)!;
    const large = calcDeploymentEstimate(MOE, offering(8), SETTINGS)!;
    // More GPUs ⇒ more usable VRAM after the fixed weights ⇒ more streams.
    expect(large.operatingStreams.high).toBeGreaterThan(small.operatingStreams.high);
  });

  it("admits at least as many streams under FP8 KV as under FP16 KV", () => {
    const fp16 = calcDeploymentEstimate(MOE, offering(8), {
      ...SETTINGS,
      kvCachePrecision: "fp16",
    })!;
    const fp8 = calcDeploymentEstimate(MOE, offering(8), {
      ...SETTINGS,
      kvCachePrecision: "auto", // H100 (Hopper) ⇒ 1-byte FP8 KV
    })!;
    expect(fp8.operatingStreams.high).toBeGreaterThan(fp16.operatingStreams.high);
    // FP8 KV halves the per-token KV footprint, so the decode-time KV-read term
    // shrinks ⇒ single-stream decode is a touch faster (or equal), never slower.
    expect(fp8.singleStreamTokS!).toBeGreaterThanOrEqual(fp16.singleStreamTokS!);
  });

  it("returns null when the layout cannot fit the weights", () => {
    // 1× H100 (80 GB) cannot hold a 235B BF16 model (~470 GB).
    expect(calcDeploymentEstimate(MOE, offering(1), SETTINGS)).toBeNull();
  });

  it("still emits streams (throughput null) when throughput dims are missing", () => {
    // hidden_size missing ⇒ throughput can't be modeled, but operating streams
    // are pure VRAM accounting and must still be produced.
    const noHidden: Model = { ...MOE, hidden_size: null };
    const est = calcDeploymentEstimate(noHidden, offering(8), SETTINGS)!;
    expect(est).not.toBeNull();
    expect(est.operatingStreams.high).toBeGreaterThanOrEqual(1);
    expect(est.singleStreamTokS).toBeNull();
    expect(est.aggregateTokS).toBeNull();
    expect(est.throughputState).toBe("data-incomplete");
  });

  it("suppresses throughput for an unsupported architecture but keeps streams", () => {
    // Linear-attention hybrid: num_kv_layers < num_hidden_layers.
    const hybrid: Model = { ...MOE, num_hidden_layers: 48, num_kv_layers: 12 };
    const est = calcDeploymentEstimate(hybrid, offering(8), SETTINGS)!;
    expect(est.operatingStreams.high).toBeGreaterThanOrEqual(1);
    expect(est.singleStreamTokS).toBeNull();
    expect(est.throughputState).toBe("unsupported-arch");
  });

  it("returns null for a GPU with no published specs", () => {
    const unknown: GpuOffering = { ...offering(8), gpu_name: "NotARealGpu" };
    expect(calcDeploymentEstimate(MOE, unknown, SETTINGS)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// DeploymentEstimate is surfaced read-only on the view data structures
// (so the result views can render it without recomputing).
// ---------------------------------------------------------------------------

describe("DeploymentEstimate surfaced on view data", () => {
  // Fully-dimensioned MoE so calcDeploymentEstimate yields a non-null estimate
  // (the swe_bench fixtures above have hidden_size === null on purpose).
  const MOE: Model = {
    model_name: "Test-MoE-235B",
    learnable_params_b: 235,
    active_params_b: 22,
    architecture: "MoE",
    context_length: 131072,
    precision: "BF16",
    routed_expert_params_b: null,
    attention_type: "GQA",
    num_hidden_layers: 94,
    hidden_size: 4096,
    num_kv_layers: null,
    num_kv_heads: 8,
    head_dim: 128,
    num_experts: 128,
    experts_per_token: 8,
    kv_lora_rank: null,
    qk_rope_head_dim: null,
    hf_model_id: null,
    model_url: null,
    kv_elems_per_token: null,
    license_name: null,
    license_url: null,
  };

  const MOE_BENCHMARK: BenchmarkScore = {
    model_name: "Test-MoE-235B",
    benchmark_name: "swe_bench_verified",
    benchmark_display_name: "SWE-bench Verified",
    score: 70.0,
    rank: 1,
    cost_per_task: null,
    benchmark_group: "coding",
    benchmark_group_display: "Coding",
  };

  const SETTINGS: typeof DEFAULT_ADVANCED_SETTINGS = {
    ...DEFAULT_ADVANCED_SETTINGS,
    avgInputTokens: 8000,
    avgOutputTokens: 1000,
  };

  const vram = getGpuThroughputSpec("H100")!.memory_size_gb;
  // 8× H100 fits the 235B BF16 weights (~470 GB) with KV headroom.
  const H100_X8: GpuOffering = {
    gpu_name: "H100",
    vram_gb: vram,
    gpu_count: 8,
    total_vram_gb: vram * 8,
    price_per_hour: 1,
    currency: "USD",
    provider: "test",
    instance_name: "test-x8",
    location: "test",
    interconnect: "nvlink",
  };

  it("calculatePerformanceMatrix attaches the estimate to each GPU setup", () => {
    const matrix = calculatePerformanceMatrix(
      [H100_X8], [MOE], [MOE_BENCHMARK], [SOTA], "swe_bench_verified", SETTINGS,
    );
    const direct = calcDeploymentEstimate(MOE, H100_X8, SETTINGS);
    expect(direct).not.toBeNull();

    const setupsWithEstimate = matrix
      .flat()
      .flatMap((cell) => cell.gpuSetups);
    expect(setupsWithEstimate.length).toBeGreaterThan(0);
    for (const setup of setupsWithEstimate) {
      // Read-only: identical to a direct compute, never re-derived in the view.
      expect(setup.deploymentEstimate).toEqual(direct);
    }
  });

  it("honors an interconnect-tier override on the offering, else uses the GPU datasheet", () => {
    // H100's datasheet tier is nvswitch (mesh) ⇒ TP across all 8, PP=1.
    const dflt = calcDeploymentEstimate(MOE, H100_X8, SETTINGS)!;
    expect(dflt.assumptions.interconnectTier).toBe("nvswitch");

    // Explicit tier enum overrides the datasheet ⇒ paired NVLink layout.
    const paired = calcDeploymentEstimate(MOE, { ...H100_X8, interconnect: "nvlink_paired" }, SETTINGS)!;
    expect(paired.assumptions.interconnectTier).toBe("nvlink_paired");

    // Legacy/descriptive strings are not tier names ⇒ fall through to datasheet.
    const legacy = calcDeploymentEstimate(MOE, { ...H100_X8, interconnect: "nvlink" }, SETTINGS)!;
    expect(legacy.assumptions.interconnectTier).toBe("nvswitch");
  });

  it("calculateBudgetChartData surfaces the estimate at the configured utilization", () => {
    const gpuConfig = {
      label: "8×H100",
      gpuName: "H100",
      gpuCount: 8,
      vramPerGpu: vram,
      totalVramGb: vram * 8,
      interconnect: "nvlink" as const,
    };
    const data = calculateBudgetChartData(
      gpuConfig, [MOE], [], [], "swe_bench_verified", DEFAULT_MEMORY_UTILIZATION, SETTINGS,
    );
    const point = data.find((d) => d.modelName === MOE.model_name)!;
    expect(point.fits).toBe(true);

    const direct = calcDeploymentEstimate(
      MOE, H100_X8, SETTINGS, DEFAULT_MEMORY_UTILIZATION,
    );
    expect(point.deploymentEstimate).toEqual(direct);
  });

  it("sizes requestsPerHour as a prefill+decode time-share on the aggregate throughput", () => {
    const gpuConfig = {
      label: "8×H100", gpuName: "H100", gpuCount: 8,
      vramPerGpu: vram, totalVramGb: vram * 8, interconnect: "nvlink" as const,
    };
    const data = calculateBudgetChartData(
      gpuConfig, [MOE], [], [], "swe_bench_verified", DEFAULT_MEMORY_UTILIZATION, SETTINGS,
    );
    const point = data.find((d) => d.modelName === MOE.model_name)!;
    const est = point.deploymentEstimate!;
    expect(est.aggregateTokS).not.toBeNull();
    expect(est.prefillComputeTokS).not.toBeNull();

    // 90% prefix cache ⇒ only 10% of input is prefilled; decode runs at the
    // aggregate batched throughput; the node serializes the two phases.
    const CACHE = 0.9;
    const prefillSec = (SETTINGS.avgInputTokens * (1 - CACHE)) / est.prefillComputeTokS!;
    const decodeSec = SETTINGS.avgOutputTokens / est.aggregateTokS!;
    expect(point.requestsPerHour!).toBeCloseTo(3600 / (prefillSec + decodeSec), 3);

    // Uses the aggregate throughput, not N × single-stream (materially higher).
    const nTimesSingle =
      (point.maxConcurrentStreams * est.singleStreamTokS!) / SETTINGS.avgOutputTokens * 3600;
    expect(point.requestsPerHour!).toBeGreaterThan(nTimesSingle);
  });

  it("flags unmodeled architectures via throughputModeled, independent of fit", () => {
    const gpuConfig = {
      label: "8×H100", gpuName: "H100", gpuCount: 8,
      vramPerGpu: vram, totalVramGb: vram * 8, interconnect: "nvlink" as const,
    };
    const data = calculateBudgetChartData(
      gpuConfig, [MOE, MINIMAX_M25], [], [], "swe_bench_verified", DEFAULT_MEMORY_UTILIZATION, SETTINGS,
    );
    const moe = data.find((d) => d.modelName === MOE.model_name)!;
    const mm = data.find((d) => d.modelName === MINIMAX_M25.model_name)!;

    // MOE is fully dimensioned → modeled; MiniMax-M2.5 lacks hidden_size → not.
    expect(moe.throughputModeled).toBe(true);
    expect(mm.throughputModeled).toBe(false);
    // The unmodeled model still sizes (fits, streams) — it's just off the chart.
    expect(mm.fits).toBe(true);
    expect(mm.requestsPerHour).toBeNull();
  });

  it("leaves the estimate null for a model that does not fit", () => {
    const gpuConfig = {
      label: "1×H100",
      gpuName: "H100",
      gpuCount: 1,
      vramPerGpu: vram,
      totalVramGb: vram,
      interconnect: null,
    };
    const data = calculateBudgetChartData(
      gpuConfig, [MOE], [], [], "swe_bench_verified", DEFAULT_MEMORY_UTILIZATION, SETTINGS,
    );
    const point = data.find((d) => d.modelName === MOE.model_name)!;
    expect(point.fits).toBe(false);
    expect(point.deploymentEstimate).toBeNull();
  });
});
