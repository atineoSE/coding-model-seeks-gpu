import { describe, it, expect } from "vitest";
import type { Model, BenchmarkScore, DeploymentEstimate } from "@/types";
import { scoreFor, isUnranked, minVramForModel, isMoE, modelArchShape } from "./model-data";
import modelsData from "../../public/data/models.json";
import {
  getModelMemory,
  resolveModelPrecision,
  WEIGHT_OVERHEAD_FACTOR,
} from "./calculations";

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
    hidden_size: null,
    num_kv_layers: null,
    num_kv_heads: null,
    head_dim: null,
    num_experts: null,
    experts_per_token: null,
    kv_lora_rank: null,
    qk_rope_head_dim: null,
    kv_elems_per_token: null,
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

// ---------------------------------------------------------------------------
// scoreFor
// ---------------------------------------------------------------------------

describe("scoreFor", () => {
  const model = makeModel({ model_name: "ModelA" });
  const benchmarks = [
    makeBenchmark("ModelA", 60, "overall"),
    makeBenchmark("ModelA", 55, "frontend"),
    makeBenchmark("ModelB", 90, "overall"),
  ];

  it("returns the matching benchmark entry for a ranked model", () => {
    const b = scoreFor(model, "overall", benchmarks);
    expect(b).not.toBeNull();
    expect(b?.score).toBe(60);
    expect(b?.benchmark_name).toBe("overall");
  });

  it("matches on the exact model_name join (no alias resolution)", () => {
    // An alias-style name must NOT resolve — the join is exact, like
    // matrix-calculator.ts.
    const aliased = makeModel({ model_name: "modela" });
    expect(scoreFor(aliased, "overall", benchmarks)).toBeNull();
  });

  it("is category-specific: ranked on overall, unranked on a category it lacks", () => {
    expect(scoreFor(model, "testing", benchmarks)).toBeNull();
    expect(scoreFor(model, "frontend", benchmarks)?.score).toBe(55);
  });

  it("returns null for a model absent from benchmarks", () => {
    const absent = makeModel({ model_name: "Nowhere" });
    expect(scoreFor(absent, "overall", benchmarks)).toBeNull();
  });

  it("treats a null score as no score (not a match)", () => {
    const nullScored = [makeBenchmark("ModelA", null, "overall")];
    expect(scoreFor(model, "overall", nullScored)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// isUnranked
// ---------------------------------------------------------------------------

describe("isUnranked", () => {
  const model = makeModel({ model_name: "ModelA" });

  it("is false when the model has a score in the category", () => {
    const benchmarks = [makeBenchmark("ModelA", 60, "overall")];
    expect(isUnranked(model, "overall", benchmarks)).toBe(false);
  });

  it("is true when the model is absent from benchmarks", () => {
    expect(isUnranked(model, "overall", [])).toBe(true);
  });

  it("is true when the category entry has a null score", () => {
    const benchmarks = [makeBenchmark("ModelA", null, "overall")];
    expect(isUnranked(model, "overall", benchmarks)).toBe(true);
  });

  it("is true for a category the model is not ranked on", () => {
    const benchmarks = [makeBenchmark("ModelA", 60, "overall")];
    expect(isUnranked(model, "frontend", benchmarks)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// minVramForModel
// ---------------------------------------------------------------------------

describe("minVramForModel", () => {
  it("returns a positive whole-GB VRAM for a sized model", () => {
    // Dense BF16, 100B params → 200 GB raw → ceil(200 * 1.15) = 230 GB.
    const model = makeModel({ model_name: "Sized", learnable_params_b: 100, precision: "BF16" });
    const expected = Math.ceil(
      getModelMemory(model, resolveModelPrecision(model))! * WEIGHT_OVERHEAD_FACTOR,
    );
    const vram = minVramForModel(model);
    expect(vram).toBe(expected);
    expect(vram).toBeGreaterThan(0);
  });

  it("returns null when sizing is unknown (learnable_params_b is null)", () => {
    const model = makeModel({ model_name: "Unsized", learnable_params_b: null });
    expect(minVramForModel(model)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// isMoE / modelArchShape
// ---------------------------------------------------------------------------

describe("isMoE", () => {
  it("is true for MoE and false for Dense", () => {
    expect(isMoE(makeModel({ model_name: "M", architecture: "MoE" }))).toBe(true);
    expect(isMoE(makeModel({ model_name: "D", architecture: "Dense" }))).toBe(false);
  });
});

describe("modelArchShape", () => {
  it("passes through MoE expert counts and hidden size", () => {
    const m = makeModel({
      model_name: "Sparse",
      architecture: "MoE",
      hidden_size: 6144,
      num_experts: 160,
      experts_per_token: 8,
    });
    expect(modelArchShape(m)).toEqual({
      hiddenSize: 6144,
      numExperts: 160,
      expertsPerToken: 8,
    });
  });

  it("nulls expert counts for dense models even if data sets them", () => {
    const m = makeModel({
      model_name: "Dense",
      architecture: "Dense",
      hidden_size: 4096,
      num_experts: 8,
      experts_per_token: 2,
    });
    expect(modelArchShape(m)).toEqual({
      hiddenSize: 4096,
      numExperts: null,
      expertsPerToken: null,
    });
  });

  it("is null-safe when fields are missing", () => {
    const m = makeModel({ model_name: "Bare", architecture: "MoE" });
    expect(modelArchShape(m)).toEqual({
      hiddenSize: null,
      numExperts: null,
      expertsPerToken: null,
    });
  });
});

// ---------------------------------------------------------------------------
// models.json data shape (new architecture fields)
// ---------------------------------------------------------------------------

describe("models.json", () => {
  const models = modelsData as unknown as Model[];

  it("parses and every entry exposes the new architecture fields (null-safe)", () => {
    expect(models.length).toBeGreaterThan(0);
    for (const m of models) {
      expect("hidden_size" in m).toBe(true);
      expect("num_experts" in m).toBe(true);
      expect("experts_per_token" in m).toBe(true);
      // MLA latent fields are present on every entry.
      expect("kv_lora_rank" in m).toBe(true);
      expect("qk_rope_head_dim" in m).toBe(true);
    }
  });

  it("populates hidden_size/experts for at least one real MoE model", () => {
    const populated = models.filter((m) => m.hidden_size !== null);
    expect(populated.length).toBeGreaterThan(0);
    for (const m of populated) {
      expect(m.hidden_size).toBeGreaterThan(0);
      if (m.architecture === "MoE" && m.num_experts !== null) {
        expect(m.num_experts).toBeGreaterThan(0);
        expect(m.experts_per_token).not.toBeNull();
        expect(m.experts_per_token!).toBeGreaterThan(0);
        expect(m.experts_per_token!).toBeLessThanOrEqual(m.num_experts);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// DeploymentEstimate type loads and is constructible
// ---------------------------------------------------------------------------

describe("DeploymentEstimate type", () => {
  it("is exported and structurally usable", () => {
    const estimate: DeploymentEstimate = {
      singleStreamTokS: 42,
      operatingStreams: { low: 1, high: 8 },
      aggregateTokS: 300,
      throughputState: "modeled",
      assumptions: {
        context: { avgInputTokens: 50_000, avgOutputTokens: 1000, prefixReuse: 0.5 },
        interconnectTier: "nvswitch",
        moe: true,
      },
    };
    expect(estimate.operatingStreams.high).toBeGreaterThanOrEqual(estimate.operatingStreams.low);
    expect(estimate.assumptions.interconnectTier).toBe("nvswitch");
  });
});
