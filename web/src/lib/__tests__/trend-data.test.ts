import { describe, it, expect } from "vitest";
import type { BenchmarkScore, SotaScore } from "@/types";
import {
  resolveModelName,
  isOpenSourceModel,
  computeGapTrend,
  deduplicateGapTrend,
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
// computeGapTrend
// ---------------------------------------------------------------------------

describe("computeGapTrend", () => {
  const openSourceNames = new Set(["ModelOpen"]);

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

  function makeSnapshot(
    date: string,
    openScore: number,
    closedScore: number,
    category = "overall",
  ): SnapshotData {
    return {
      date,
      benchmarks: [
        makeBenchmark("ModelOpen", openScore, category),
        makeBenchmark("claude-opus-4-6", closedScore, category),
      ],
      sotaScores: [],
    };
  }

  it("returns one point per snapshot with both closed and open scores", () => {
    const snapshots = [
      makeSnapshot("2026-01-28", 40, 60),
      makeSnapshot("2026-02-01", 50, 62),
    ];
    const result = computeGapTrend(snapshots, openSourceNames, "overall");
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({
      date: "2026-01-28",
      closedSourceScore: 60,
      closedSourceModel: "claude-opus-4-6",
      openSourceScore: 40,
      openSourceModel: "ModelOpen",
    });
    expect(result[1].openSourceScore).toBe(50);
  });

  it("skips snapshots missing the selected category", () => {
    const snapshots = [makeSnapshot("2026-01-28", 40, 60)];
    const result = computeGapTrend(snapshots, openSourceNames, "frontend");
    expect(result).toHaveLength(0);
  });

  it("skips snapshots with only one side", () => {
    const oneSided: SnapshotData[] = [
      {
        date: "2026-02-01",
        benchmarks: [makeBenchmark("ModelOpen", 55, "overall")],
        sotaScores: [],
      },
    ];
    const result = computeGapTrend(oneSided, openSourceNames, "overall");
    expect(result).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// deduplicateGapTrend
// ---------------------------------------------------------------------------

describe("deduplicateGapTrend", () => {
  function point(
    date: string,
    closedScore: number,
    closedModel: string,
    openScore: number,
    openModel: string,
  ): GapTrendPoint {
    return {
      date,
      closedSourceScore: closedScore,
      closedSourceModel: closedModel,
      openSourceScore: openScore,
      openSourceModel: openModel,
    };
  }

  it("returns empty array for empty input", () => {
    expect(deduplicateGapTrend([])).toEqual([]);
  });

  it("returns the single point for a one-element array", () => {
    const input = [point("2026-01-01", 60, "A", 40, "B")];
    expect(deduplicateGapTrend(input)).toEqual(input);
  });

  it("keeps first and removes consecutive duplicates", () => {
    const input = [
      point("2026-01-01", 60, "A", 40, "B"),
      point("2026-01-02", 60, "A", 40, "B"),
      point("2026-01-03", 60, "A", 40, "B"),
    ];
    const result = deduplicateGapTrend(input);
    // First kept, rest identical so dropped
    expect(result).toHaveLength(1);
    expect(result[0].date).toBe("2026-01-01");
  });

  it("keeps points where closed score changes", () => {
    const input = [
      point("2026-01-01", 60, "A", 40, "B"),
      point("2026-01-02", 65, "A", 40, "B"),
    ];
    const result = deduplicateGapTrend(input);
    expect(result).toHaveLength(2);
  });

  it("keeps points where open model changes", () => {
    const input = [
      point("2026-01-01", 60, "A", 40, "B"),
      point("2026-01-02", 60, "A", 40, "C"),
    ];
    const result = deduplicateGapTrend(input);
    expect(result).toHaveLength(2);
  });

  it("drops trailing duplicate after a change", () => {
    const input = [
      point("2026-01-01", 60, "A", 40, "B"),
      point("2026-01-02", 65, "A2", 45, "B2"),
      point("2026-01-03", 65, "A2", 45, "B2"),
    ];
    const result = deduplicateGapTrend(input);
    expect(result).toHaveLength(2);
    expect(result[1].date).toBe("2026-01-02");
  });

  it("does not duplicate last point if it is already a change-point", () => {
    const input = [
      point("2026-01-01", 60, "A", 40, "B"),
      point("2026-01-02", 65, "A2", 45, "B2"),
    ];
    const result = deduplicateGapTrend(input);
    expect(result).toHaveLength(2);
  });
});
