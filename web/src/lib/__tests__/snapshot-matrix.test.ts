import { describe, it, expect } from "vitest";
import type { BenchmarkScore } from "@/types";
import {
  findBestModelsPerLab,
  getMatrixModels,
  BENCHMARK_CATEGORIES,
} from "../snapshot-matrix";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeEntry(
  overrides: Partial<BenchmarkScore> & Pick<BenchmarkScore, "model_name" | "benchmark_name">,
): BenchmarkScore {
  return {
    benchmark_display_name: overrides.benchmark_name,
    score: null,
    rank: null,
    cost_per_task: null,
    benchmark_group: "openhands",
    benchmark_group_display: "OpenHands Index",
    openness: "open_weights",
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// findBestModelsPerLab
// ---------------------------------------------------------------------------

describe("findBestModelsPerLab", () => {
  it("picks the model with the highest overall score per lab", () => {
    const benchmarks: BenchmarkScore[] = [
      makeEntry({ model_name: "claude-opus-4-6", benchmark_name: "overall", score: 66.7, openness: "closed_api_available" }),
      makeEntry({ model_name: "claude-opus-4-5", benchmark_name: "overall", score: 60.6, openness: "closed_api_available" }),
      makeEntry({ model_name: "GPT-5.4", benchmark_name: "overall", score: 63.8, openness: "closed_api_available" }),
      makeEntry({ model_name: "GPT-5.2", benchmark_name: "overall", score: 56.3, openness: "closed_api_available" }),
      makeEntry({ model_name: "Gemini-3.1-Pro", benchmark_name: "overall", score: 55.7, openness: "closed_api_available" }),
    ];

    const result = findBestModelsPerLab(benchmarks);

    expect(result).toEqual({
      anthropic: "claude-opus-4-6",
      openai: "GPT-5.4",
      google: "Gemini-3.1-Pro",
    });
  });

  it("ignores open_weights models", () => {
    const benchmarks: BenchmarkScore[] = [
      makeEntry({ model_name: "claude-opus-4-6", benchmark_name: "overall", score: 66.7, openness: "open_weights" }),
    ];

    expect(findBestModelsPerLab(benchmarks)).toEqual({});
  });

  it("ignores models not in MODEL_LAB_MAP", () => {
    const benchmarks: BenchmarkScore[] = [
      makeEntry({ model_name: "unknown-model", benchmark_name: "overall", score: 99, openness: "closed_api_available" }),
    ];

    expect(findBestModelsPerLab(benchmarks)).toEqual({});
  });

  it("prefers overall score over non-overall", () => {
    const benchmarks: BenchmarkScore[] = [
      // Non-overall entry with high score
      makeEntry({ model_name: "GPT-5.4", benchmark_name: "frontend", score: 90.0, openness: "closed_api_available" }),
      // Overall entry with lower score
      makeEntry({ model_name: "GPT-5.4", benchmark_name: "overall", score: 60.0, openness: "closed_api_available" }),
      // Another model with higher non-overall score
      makeEntry({ model_name: "GPT-5.2", benchmark_name: "frontend", score: 95.0, openness: "closed_api_available" }),
    ];

    const result = findBestModelsPerLab(benchmarks);
    // GPT-5.4 has overall, GPT-5.2 does not — GPT-5.4 wins
    expect(result.openai).toBe("GPT-5.4");
  });

  it("falls back to non-overall score when no overall exists", () => {
    const benchmarks: BenchmarkScore[] = [
      makeEntry({ model_name: "GPT-5.4", benchmark_name: "frontend", score: 70.0, openness: "closed_api_available" }),
      makeEntry({ model_name: "GPT-5.2", benchmark_name: "frontend", score: 80.0, openness: "closed_api_available" }),
    ];

    const result = findBestModelsPerLab(benchmarks);
    expect(result.openai).toBe("GPT-5.2");
  });

  it("skips entries with null score", () => {
    const benchmarks: BenchmarkScore[] = [
      makeEntry({ model_name: "GPT-5.4", benchmark_name: "overall", score: null, openness: "closed_api_available" }),
    ];

    expect(findBestModelsPerLab(benchmarks)).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// getMatrixModels
// ---------------------------------------------------------------------------

describe("getMatrixModels", () => {
  const benchmarks: BenchmarkScore[] = [
    // Open model with all categories
    makeEntry({ model_name: "DeepSeek-V3", benchmark_name: "overall", score: 45.0, openness: "open_weights" }),
    makeEntry({ model_name: "DeepSeek-V3", benchmark_name: "frontend", score: 50.0, openness: "open_weights" }),
    makeEntry({ model_name: "DeepSeek-V3", benchmark_name: "greenfield", score: 40.0, openness: "open_weights" }),

    // Open model without overall
    makeEntry({ model_name: "Qwen3-Coder-480B", benchmark_name: "frontend", score: 30.0, openness: "open_weights" }),

    // Best closed model (Anthropic)
    makeEntry({ model_name: "claude-opus-4-6", benchmark_name: "overall", score: 66.7, openness: "closed_api_available" }),
    makeEntry({ model_name: "claude-opus-4-6", benchmark_name: "frontend", score: 70.0, openness: "closed_api_available" }),

    // Worse closed model (Anthropic) — should NOT appear in matrix
    makeEntry({ model_name: "claude-opus-4-5", benchmark_name: "overall", score: 60.6, openness: "closed_api_available" }),

    // Best closed model (OpenAI)
    makeEntry({ model_name: "GPT-5.4", benchmark_name: "overall", score: 63.8, openness: "closed_api_available" }),
  ];

  it("includes all open-weights models", () => {
    const models = getMatrixModels(benchmarks);
    const names = models.map((m) => m.modelName);
    expect(names).toContain("DeepSeek-V3");
    expect(names).toContain("Qwen3-Coder-480B");
  });

  it("includes only the best closed model per lab", () => {
    const models = getMatrixModels(benchmarks);
    const names = models.map((m) => m.modelName);
    expect(names).toContain("claude-opus-4-6");
    expect(names).not.toContain("claude-opus-4-5");
    expect(names).toContain("GPT-5.4");
  });

  it("sets lab for closed models and null for open models", () => {
    const models = getMatrixModels(benchmarks);
    const claude = models.find((m) => m.modelName === "claude-opus-4-6");
    const deepseek = models.find((m) => m.modelName === "DeepSeek-V3");
    expect(claude?.lab).toBe("anthropic");
    expect(deepseek?.lab).toBeNull();
  });

  it("sorts by overall score descending, models without overall last", () => {
    const models = getMatrixModels(benchmarks);
    const names = models.map((m) => m.modelName);
    // claude-opus-4-6 (66.7) > GPT-5.4 (63.8) > DeepSeek-V3 (45.0) > Qwen3-Coder-480B (no overall)
    expect(names).toEqual(["claude-opus-4-6", "GPT-5.4", "DeepSeek-V3", "Qwen3-Coder-480B"]);
  });

  it("populates scores for each benchmark", () => {
    const models = getMatrixModels(benchmarks);
    const deepseek = models.find((m) => m.modelName === "DeepSeek-V3")!;
    expect(deepseek.scores["frontend"]).toBe(50.0);
    expect(deepseek.scores["greenfield"]).toBe(40.0);
    expect(deepseek.scores["testing"]).toBeUndefined();
  });

  it("returns empty array for empty input", () => {
    expect(getMatrixModels([])).toEqual([]);
  });

  it("models without overall are sorted by average available score descending", () => {
    const entries: BenchmarkScore[] = [
      // Zebra-Model sorts first alphabetically but has lower scores
      makeEntry({ model_name: "Zebra-Model", benchmark_name: "frontend", score: 90.0, openness: "open_weights" }),
      makeEntry({ model_name: "Zebra-Model", benchmark_name: "greenfield", score: 80.0, openness: "open_weights" }),
      makeEntry({ model_name: "Alpha-Model", benchmark_name: "frontend", score: 10.0, openness: "open_weights" }),
    ];
    const models = getMatrixModels(entries);
    // Zebra-Model avg=(90+80)/2=85 > Alpha-Model avg=10, so Zebra ranks first despite alphabetical order
    expect(models.map((m) => m.modelName)).toEqual(["Zebra-Model", "Alpha-Model"]);
  });
});

// ---------------------------------------------------------------------------
// BENCHMARK_CATEGORIES
// ---------------------------------------------------------------------------

describe("BENCHMARK_CATEGORIES", () => {
  it("contains exactly the 5 expected categories", () => {
    expect(BENCHMARK_CATEGORIES).toEqual([
      "frontend",
      "greenfield",
      "issue_resolution",
      "testing",
      "information_gathering",
    ]);
  });
});
