import { describe, it, expect } from "vitest";
import type { BenchmarkScore, SotaScore, Model, GpuOffering, AdvancedSettings } from "@/types";
import {
  resolveModelName,
  isOpenSourceModel,
  computeGapTrend,
  computeCostTrend,
  computeSotaPercentTrend,
  computeModelSizeScore,
  type SnapshotData,
  type GapTrendPoint,
} from "../trend-data";
import { WEIGHT_OVERHEAD_FACTOR } from "../calculations";

// ---------------------------------------------------------------------------
// resolveModelName
// ---------------------------------------------------------------------------

describe("resolveModelName", () => {
  it("returns the same name for unknown models", () => {
    expect(resolveModelName("some-random-model")).toBe("some-random-model");
  });

  it("resolves a single rename", () => {
    expect(resolveModelName("gpt-5")).toBe("GPT-5.2");
  });

  it("chains renames", () => {
    // jade-spark-2862 → Minimax-2.5 → MiniMax-M2.5
    expect(resolveModelName("jade-spark-2862")).toBe("MiniMax-M2.5");
  });

  it("resolves claude renames", () => {
    expect(resolveModelName("claude-opus-4-5-20251101")).toBe("claude-opus-4-5");
    expect(resolveModelName("claude-4.5-opus")).toBe("claude-opus-4-5");
  });

  it("resolves multiple old names to the same canonical name", () => {
    expect(resolveModelName("minimax-m2")).toBe("MiniMax-M2.1");
    expect(resolveModelName("minimax-m2.1")).toBe("MiniMax-M2.1");
  });

  it("resolves early snapshot model names", () => {
    expect(resolveModelName("Qwen3-Coder-480B-A35B-Instruct-FP8")).toBe("Qwen3-Coder-480B");
    expect(resolveModelName("claude-sonnet-4-5-20250929")).toBe("claude-sonnet-4-5");
  });

  it("does not modify already-canonical names", () => {
    expect(resolveModelName("DeepSeek-V3.2-Reasoner")).toBe("DeepSeek-V3.2-Reasoner");
  });

  it("resolves bare 'nemotron' alias", () => {
    expect(resolveModelName("nemotron")).toBe("Nemotron-3-Nano");
  });
});

// ---------------------------------------------------------------------------
// isOpenSourceModel
// ---------------------------------------------------------------------------

describe("isOpenSourceModel", () => {
  const openSourceNames = new Set([
    "DeepSeek-V3.2-Reasoner",
    "Qwen3-Coder-480B",
    "MiniMax-M2.5",
  ]);

  it("returns true for a model in the open-source list", () => {
    expect(isOpenSourceModel("DeepSeek-V3.2-Reasoner", openSourceNames)).toBe(true);
  });

  it("returns true for an alias that resolves to an open-source model", () => {
    expect(isOpenSourceModel("deepseek-v3.2-reasoner", openSourceNames)).toBe(true);
    expect(isOpenSourceModel("jade-spark-2862", openSourceNames)).toBe(true);
  });

  it("returns false for closed-source models", () => {
    expect(isOpenSourceModel("claude-opus-4-6", openSourceNames)).toBe(false);
    expect(isOpenSourceModel("GPT-5.2", openSourceNames)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// computeGapTrend (roster-based)
// ---------------------------------------------------------------------------

describe("computeGapTrend", () => {
  const openSourceNames = new Set(["ModelOpen", "ModelOpen2"]);

  function makeBenchmark(
    model_name: string,
    score: number,
    benchmark_name = "overall",
  ): BenchmarkScore {
    return {
      model_name,
      benchmark_name,
      benchmark_display_name: benchmark_name === "overall" ? "Overall" : benchmark_name,
      score,
      rank: null,
      cost_per_task: null,
      benchmark_group: "openhands",
      benchmark_group_display: "OpenHands Index",
    };
  }

  it("produces one point for the initial snapshot", () => {
    const snapshots: SnapshotData[] = [
      {
        date: "2026-01-28",
        benchmarks: [
          makeBenchmark("ClosedA", 60),
          makeBenchmark("ModelOpen", 40),
        ],
        sotaScores: [],
      },
    ];
    const result = computeGapTrend(snapshots, openSourceNames, "overall");
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({
      date: "2026-01-28",
      closedSourceScore: 60,
      closedSourceModel: "ClosedA",
      openSourceScore: 40,
      openSourceModel: "ModelOpen",
    });
  });

  it("emits a new point when a new model takes the lead", () => {
    // Two snapshots — ModelOpen2 appears in snap2 with a higher latest score
    const snapshots: SnapshotData[] = [
      {
        date: "2026-01-28",
        benchmarks: [
          makeBenchmark("ClosedA", 60),
          makeBenchmark("ModelOpen", 40),
        ],
        sotaScores: [],
      },
      {
        date: "2026-02-05",
        benchmarks: [
          makeBenchmark("ClosedA", 60),
          makeBenchmark("ModelOpen", 40),
          makeBenchmark("ModelOpen2", 50),
        ],
        sotaScores: [],
      },
    ];
    const result = computeGapTrend(snapshots, openSourceNames, "overall");
    expect(result).toHaveLength(2);
    expect(result[1].openSourceModel).toBe("ModelOpen2");
    expect(result[1].openSourceScore).toBe(50);
    expect(result[1].date).toBe("2026-02-05");
  });

  it("does NOT emit when a new model appears but does not take the lead", () => {
    // ModelOpen2 appears but with a lower score than ModelOpen
    const snapshots: SnapshotData[] = [
      {
        date: "2026-01-28",
        benchmarks: [
          makeBenchmark("ClosedA", 60),
          makeBenchmark("ModelOpen", 40),
        ],
        sotaScores: [],
      },
      {
        date: "2026-02-05",
        benchmarks: [
          makeBenchmark("ClosedA", 60),
          makeBenchmark("ModelOpen", 40),
          makeBenchmark("ModelOpen2", 30),
        ],
        sotaScores: [],
      },
    ];
    const result = computeGapTrend(snapshots, openSourceNames, "overall");
    expect(result).toHaveLength(1);
  });

  it("does NOT emit for score recalculations (same roster)", () => {
    // Same models in both snapshots — just score changes
    const snapshots: SnapshotData[] = [
      {
        date: "2026-01-28",
        benchmarks: [
          makeBenchmark("ClosedA", 58),
          makeBenchmark("ModelOpen", 38),
        ],
        sotaScores: [],
      },
      {
        date: "2026-02-01",
        benchmarks: [
          makeBenchmark("ClosedA", 60),
          makeBenchmark("ModelOpen", 40),
        ],
        sotaScores: [],
      },
    ];
    const result = computeGapTrend(snapshots, openSourceNames, "overall");
    // Only the first point — snap2 adds no new models
    expect(result).toHaveLength(1);
  });

  it("uses latest scores, not per-snapshot scores", () => {
    // ModelOpen had 38 in snap1 but latest (snap2) says 40
    const snapshots: SnapshotData[] = [
      {
        date: "2026-01-28",
        benchmarks: [
          makeBenchmark("ClosedA", 58),
          makeBenchmark("ModelOpen", 38),
        ],
        sotaScores: [],
      },
      {
        date: "2026-02-01",
        benchmarks: [
          makeBenchmark("ClosedA", 60),
          makeBenchmark("ModelOpen", 40),
        ],
        sotaScores: [],
      },
    ];
    const result = computeGapTrend(snapshots, openSourceNames, "overall");
    // First point uses latest score (40), not snapshot score (38)
    expect(result[0].openSourceScore).toBe(40);
    expect(result[0].closedSourceScore).toBe(60);
  });

  it("sets point date to first-appearance of the newer leader", () => {
    // ClosedB appears in snap2, takes closed lead
    // Its first-seen date should be the point date
    const snapshots: SnapshotData[] = [
      {
        date: "2026-01-28",
        benchmarks: [
          makeBenchmark("ClosedA", 50),
          makeBenchmark("ModelOpen", 40),
        ],
        sotaScores: [],
      },
      {
        date: "2026-02-05",
        benchmarks: [
          makeBenchmark("ClosedA", 50),
          makeBenchmark("ClosedB", 70),
          makeBenchmark("ModelOpen", 40),
        ],
        sotaScores: [],
      },
    ];
    const result = computeGapTrend(snapshots, openSourceNames, "overall");
    expect(result).toHaveLength(2);
    expect(result[1].closedSourceModel).toBe("ClosedB");
    // ClosedB first seen in snap2, ModelOpen first seen in snap1 → newer is ClosedB
    expect(result[1].date).toBe("2026-02-05");
  });

  it("skips snapshots missing the selected category", () => {
    const snapshots: SnapshotData[] = [
      {
        date: "2026-01-28",
        benchmarks: [
          makeBenchmark("ClosedA", 60),
          makeBenchmark("ModelOpen", 40),
        ],
        sotaScores: [],
      },
    ];
    const result = computeGapTrend(snapshots, openSourceNames, "frontend");
    expect(result).toHaveLength(0);
  });

  it("returns empty for empty snapshots", () => {
    expect(computeGapTrend([], openSourceNames, "overall")).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// computeCostTrend
// ---------------------------------------------------------------------------

describe("computeCostTrend", () => {
  const TEST_MODEL: Model = {
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

  const TEST_GPU: GpuOffering = {
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

  const TEST_SETTINGS: AdvancedSettings = {
    avgInputTokens: 40_000,
    avgOutputTokens: 1500,
    minTokPerStream: 20,
  };

  it("produces matching cost points from gap trend points", () => {
    const gapPoints: GapTrendPoint[] = [
      {
        date: "2026-01-28",
        closedSourceScore: 60,
        closedSourceModel: "ClosedA",
        openSourceScore: 44.4,
        openSourceModel: "MiniMax-M2.5",
      },
    ];

    const result = computeCostTrend(gapPoints, [TEST_MODEL], [TEST_GPU], TEST_SETTINGS);
    expect(result).toHaveLength(1);
    expect(result[0].date).toBe("2026-01-28");
    expect(result[0].modelName).toBe("MiniMax-M2.5");
    expect(result[0].score).toBe(44.4);
    expect(result[0].monthlyCost).toBeGreaterThan(0);
  });

  it("skips points where model is not found", () => {
    const gapPoints: GapTrendPoint[] = [
      {
        date: "2026-01-28",
        closedSourceScore: 60,
        closedSourceModel: "ClosedA",
        openSourceScore: 40,
        openSourceModel: "NonexistentModel",
      },
    ];

    const result = computeCostTrend(gapPoints, [], [], TEST_SETTINGS);
    expect(result).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// computeSotaPercentTrend
// ---------------------------------------------------------------------------

describe("computeSotaPercentTrend", () => {
  it("computes the ratio when the closed leader is ahead", () => {
    const gapPoints: GapTrendPoint[] = [
      {
        date: "2026-01-28",
        closedSourceScore: 80,
        closedSourceModel: "ClosedA",
        openSourceScore: 40,
        openSourceModel: "OpenA",
      },
    ];
    const result = computeSotaPercentTrend(gapPoints);
    expect(result).toHaveLength(1);
    expect(result[0].date).toBe("2026-01-28");
    expect(result[0].openSourceModel).toBe("OpenA");
    expect(result[0].percentOfSota).toBeCloseTo(0.5, 6);
  });

  it("caps at 1.0 when open ≥ closed", () => {
    const gapPoints: GapTrendPoint[] = [
      {
        date: "2026-02-01",
        closedSourceScore: 60,
        closedSourceModel: "ClosedA",
        openSourceScore: 75,
        openSourceModel: "OpenLeads",
      },
      {
        date: "2026-02-02",
        closedSourceScore: 60,
        closedSourceModel: "ClosedA",
        openSourceScore: 60,
        openSourceModel: "OpenTies",
      },
    ];
    const result = computeSotaPercentTrend(gapPoints);
    expect(result[0].percentOfSota).toBe(1);
    expect(result[1].percentOfSota).toBe(1);
  });

  it("preserves dates and model names from each input point", () => {
    const gapPoints: GapTrendPoint[] = [
      {
        date: "2026-01-28",
        closedSourceScore: 100,
        closedSourceModel: "ClosedA",
        openSourceScore: 30,
        openSourceModel: "OpenA",
      },
      {
        date: "2026-02-15",
        closedSourceScore: 100,
        closedSourceModel: "ClosedA",
        openSourceScore: 60,
        openSourceModel: "OpenB",
      },
    ];
    const result = computeSotaPercentTrend(gapPoints);
    expect(result.map((p) => p.date)).toEqual(["2026-01-28", "2026-02-15"]);
    expect(result.map((p) => p.openSourceModel)).toEqual(["OpenA", "OpenB"]);
    expect(result[0].percentOfSota).toBeCloseTo(0.3, 6);
    expect(result[1].percentOfSota).toBeCloseTo(0.6, 6);
  });

  it("returns empty for empty input", () => {
    expect(computeSotaPercentTrend([])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// computeModelSizeScore
// ---------------------------------------------------------------------------

describe("computeModelSizeScore", () => {
  function makeModel(overrides: Partial<Model> & { model_name: string }): Model {
    return {
      learnable_params_b: 100,
      active_params_b: null,
      architecture: "Dense",
      context_length: 131072,
      precision: "BF16",
      routed_expert_params_b: null,
      attention_type: "GQA",
      num_hidden_layers: null,
      num_kv_layers: null,
      num_kv_heads: null,
      head_dim: null,
      kv_lora_rank: null,
      qk_rope_head_dim: null,
      hf_model_id: null,
      model_url: null,
      license_name: null,
      license_url: null,
      ...overrides,
    };
  }

  function makeBenchmark(
    model_name: string,
    score: number | null,
    benchmark_name = "overall",
  ): BenchmarkScore {
    return {
      model_name,
      benchmark_name,
      benchmark_display_name: benchmark_name === "overall" ? "Overall" : benchmark_name,
      score,
      rank: null,
      cost_per_task: null,
      benchmark_group: "openhands",
      benchmark_group_display: "OpenHands Index",
    };
  }

  it("filters to open-source models and applies WEIGHT_OVERHEAD_FACTOR", () => {
    // Dense BF16 model, 100B params → 200 GB raw → 230 GB with overhead → ceil 230
    const openModel = makeModel({
      model_name: "OpenLarge",
      learnable_params_b: 100,
      precision: "BF16",
    });
    const closedModel = makeModel({
      model_name: "ClosedX",
      learnable_params_b: 50,
      precision: "BF16",
    });
    const benchmarks = [
      makeBenchmark("OpenLarge", 45),
      makeBenchmark("ClosedX", 90),
    ];
    const openSourceNames = new Set(["OpenLarge"]);
    const result = computeModelSizeScore(
      benchmarks,
      openSourceNames,
      [openModel, closedModel],
      "overall",
    );

    expect(result).toHaveLength(1);
    expect(result[0].modelName).toBe("OpenLarge");
    expect(result[0].score).toBe(45);
    expect(result[0].params).toBe(100);
    // 100 * 2 = 200 GB raw; 200 * 1.15 = 230 GB
    const expected = Math.ceil(100 * 2 * WEIGHT_OVERHEAD_FACTOR);
    expect(result[0].minVramGb).toBe(expected);
  });

  it("drops models with missing params, missing scores, or no model entry", () => {
    const modelNoParams = makeModel({
      model_name: "NoParams",
      learnable_params_b: null,
    });
    const modelFine = makeModel({
      model_name: "Fine",
      learnable_params_b: 20,
      precision: "BF16",
    });
    const benchmarks = [
      makeBenchmark("NoParams", 50),
      makeBenchmark("Fine", 40),
      makeBenchmark("NoScore", null), // null score → ignored
      makeBenchmark("NotInModelsList", 60), // resolved but no Model entry
    ];
    const openSourceNames = new Set([
      "NoParams",
      "Fine",
      "NoScore",
      "NotInModelsList",
    ]);
    const result = computeModelSizeScore(
      benchmarks,
      openSourceNames,
      [modelNoParams, modelFine],
      "overall",
    );

    // Only "Fine" survives
    expect(result).toHaveLength(1);
    expect(result[0].modelName).toBe("Fine");
  });

  it("picks the highest score per model when duplicates exist", () => {
    const model = makeModel({
      model_name: "Dup",
      learnable_params_b: 10,
      precision: "BF16",
    });
    const benchmarks = [
      makeBenchmark("Dup", 30),
      makeBenchmark("Dup", 55), // winner
      makeBenchmark("Dup", 42),
    ];
    const result = computeModelSizeScore(
      benchmarks,
      new Set(["Dup"]),
      [model],
      "overall",
    );
    expect(result).toHaveLength(1);
    expect(result[0].score).toBe(55);
  });

  it("resolves aliases before matching against open-source names", () => {
    // "jade-spark-2862" → "MiniMax-M2.5" via alias chain
    const model = makeModel({
      model_name: "MiniMax-M2.5",
      learnable_params_b: 228.7,
      precision: "FP8",
    });
    const benchmarks = [makeBenchmark("jade-spark-2862", 44)];
    const result = computeModelSizeScore(
      benchmarks,
      new Set(["MiniMax-M2.5"]),
      [model],
      "overall",
    );
    expect(result).toHaveLength(1);
    expect(result[0].modelName).toBe("MiniMax-M2.5");
    expect(result[0].score).toBe(44);
  });

  it("skips benchmarks in other categories", () => {
    const model = makeModel({
      model_name: "Fine",
      learnable_params_b: 10,
      precision: "BF16",
    });
    const benchmarks = [
      makeBenchmark("Fine", 30, "overall"),
      makeBenchmark("Fine", 99, "frontend"),
    ];
    const result = computeModelSizeScore(
      benchmarks,
      new Set(["Fine"]),
      [model],
      "overall",
    );
    expect(result).toHaveLength(1);
    expect(result[0].score).toBe(30);
  });
});
