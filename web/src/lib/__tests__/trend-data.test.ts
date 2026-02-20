import { describe, it, expect } from "vitest";
import type { BenchmarkScore, SotaScore, Model, GpuOffering, AdvancedSettings } from "@/types";
import {
  resolveModelName,
  isOpenSourceModel,
  computeGapTrend,
  computeCostTrend,
  type SnapshotData,
  type GapTrendPoint,
} from "../trend-data";

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
    avgInputTokens: 4000,
    avgOutputTokens: 1500,
    minTokPerStream: 20,
    prefixCacheHitRate: 80,
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
