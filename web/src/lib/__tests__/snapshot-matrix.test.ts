import { describe, it, expect } from "vitest";
import type { BenchmarkScore, Model } from "@/types";
import {
  findBestModelsPerLab,
  getMatrixModels,
  BENCHMARK_CATEGORIES,
} from "../snapshot-matrix";

function makeModel(name: string): Model {
  return {
    model_name: name,
    learnable_params_b: 100,
    active_params_b: 10,
    architecture: "MoE",
    context_length: 128000,
    precision: "BF16",
    routed_expert_params_b: null,
    attention_type: "GQA",
    num_hidden_layers: 60,
    hidden_size: null,
    num_kv_layers: null,
    num_kv_heads: 8,
    head_dim: 128,
    num_experts: null,
    experts_per_token: null,
    kv_lora_rank: null,
    qk_rope_head_dim: null,
    kv_elems_per_token: null,
    hf_model_id: `org/${name}`,
    model_url: null,
    license_name: "MIT",
    license_url: "https://example.com/license",
  };
}

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

  it("ignores models with unrecognised name prefix", () => {
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

  it("override wins over the score-derived selection", () => {
    const benchmarks: BenchmarkScore[] = [
      makeEntry({ model_name: "claude-opus-4-6", benchmark_name: "overall", score: 66.7, openness: "closed_api_available" }),
      makeEntry({ model_name: "GPT-5.4", benchmark_name: "overall", score: 63.8, openness: "closed_api_available" }),
    ];

    const result = findBestModelsPerLab(benchmarks, { anthropic: "claude-opus-5" });
    expect(result.anthropic).toBe("claude-opus-5"); // pinned
    expect(result.openai).toBe("GPT-5.4"); // untouched
  });

  it("override injects a lab absent from benchmarks", () => {
    const benchmarks: BenchmarkScore[] = [
      makeEntry({ model_name: "GPT-5.4", benchmark_name: "overall", score: 63.8, openness: "closed_api_available" }),
    ];

    const result = findBestModelsPerLab(benchmarks, { anthropic: "claude-fable-5" });
    expect(result.anthropic).toBe("claude-fable-5");
    expect(result.openai).toBe("GPT-5.4");
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

  it("marks benchmarked models as ranked (unranked=false)", () => {
    const models = getMatrixModels(benchmarks);
    const deepseek = models.find((m) => m.modelName === "DeepSeek-V3")!;
    const claude = models.find((m) => m.modelName === "claude-opus-4-6")!;
    const qwen = models.find((m) => m.modelName === "Qwen3-Coder-480B")!;
    expect(deepseek.unranked).toBe(false);
    expect(claude.unranked).toBe(false);
    // Has a frontend score but no overall — still ranked.
    expect(qwen.unranked).toBe(false);
  });

  it("surfaces models.json open-weights models with no snapshot entry as unranked", () => {
    const models = getMatrixModels(benchmarks, [
      makeModel("GLM-5.2"),
      makeModel("DeepSeek-V3"), // already ranked via benchmarks — must not duplicate
    ]);

    const glm = models.filter((m) => m.modelName === "GLM-5.2");
    expect(glm).toHaveLength(1);
    expect(glm[0].unranked).toBe(true);
    expect(glm[0].lab).toBeNull();
    expect(glm[0].scores["frontend"]).toBeUndefined();

    // DeepSeek-V3 stays ranked and is not duplicated by the models.json entry.
    const deepseek = models.filter((m) => m.modelName === "DeepSeek-V3");
    expect(deepseek).toHaveLength(1);
    expect(deepseek[0].unranked).toBe(false);
  });

  it("sorts unranked models (no score) to the bottom", () => {
    const models = getMatrixModels(benchmarks, [makeModel("GLM-5.2")]);
    expect(models[models.length - 1].modelName).toBe("GLM-5.2");
  });

  it("pins a benchmarked closed model over the score-max default", () => {
    const models = getMatrixModels(benchmarks, [], { anthropic: "claude-opus-4-5" });
    const names = models.map((m) => m.modelName);
    expect(names).toContain("claude-opus-4-5");
    expect(names).not.toContain("claude-opus-4-6"); // score-max default is replaced
    const pinned = models.find((m) => m.modelName === "claude-opus-4-5")!;
    expect(pinned.lab).toBe("anthropic");
    expect(pinned.unranked).toBe(false); // it has benchmark scores
  });

  it("surfaces an injected closed model with no benchmark entry as an unranked closed row", () => {
    const models = getMatrixModels(benchmarks, [], { anthropic: "claude-fable-5" });
    const injected = models.filter((m) => m.modelName === "claude-fable-5");
    expect(injected).toHaveLength(1);
    expect(injected[0].lab).toBe("anthropic"); // shows in the closed section
    expect(injected[0].unranked).toBe(true); // no scores until OpenHands lands
    expect(injected[0].scores["frontend"]).toBeUndefined();
    // Injected (unranked) closed model sorts to the bottom.
    expect(models[models.length - 1].modelName).toBe("claude-fable-5");
    // The score-derived anthropic model is replaced, not shown alongside.
    expect(models.map((m) => m.modelName)).not.toContain("claude-opus-4-6");
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
