import { describe, it, expect } from "vitest";
import type { Model } from "@/types";
import {
  throughputState,
  isThroughputModeled,
  isLinearAttentionHybrid,
} from "@/lib/throughput-support";
import modelsData from "../../../public/data/models.json";

/** Minimal valid Model; override per case. Defaults to a standard GQA MoE. */
function model(overrides: Partial<Model> = {}): Model {
  return {
    model_name: "Test",
    learnable_params_b: 200,
    active_params_b: 10,
    architecture: "MoE",
    context_length: 131072,
    precision: "FP8",
    routed_expert_params_b: 190,
    attention_type: "GQA",
    num_hidden_layers: 60,
    hidden_size: 4096,
    num_kv_layers: null,
    num_kv_heads: 8,
    head_dim: 128,
    num_experts: 128,
    experts_per_token: 8,
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

describe("throughputState", () => {
  it("standard GQA MoE with dims → modeled", () => {
    expect(throughputState(model({ attention_type: "GQA" }))).toBe("modeled");
  });

  it("standard MLA MoE with dims → modeled", () => {
    expect(
      throughputState(model({ attention_type: "MLA", kv_lora_rank: 512, qk_rope_head_dim: 64 })),
    ).toBe("modeled");
  });

  it("linear-attention hybrid (num_kv_layers < num_hidden_layers) → unsupported-arch", () => {
    expect(throughputState(model({ num_hidden_layers: 48, num_kv_layers: 12 }))).toBe(
      "unsupported-arch",
    );
  });

  it("sparse attention DSV4 / MSA → unsupported-arch", () => {
    expect(throughputState(model({ attention_type: "DSV4", kv_elems_per_token: 4924 }))).toBe(
      "unsupported-arch",
    );
    expect(throughputState(model({ attention_type: "MSA", kv_elems_per_token: 68736 }))).toBe(
      "unsupported-arch",
    );
  });

  it("supported architecture missing hidden_size → data-incomplete", () => {
    expect(throughputState(model({ hidden_size: null }))).toBe("data-incomplete");
  });

  it("architecture reason wins over a data gap", () => {
    // hybrid AND missing hidden_size → still unsupported-arch (won't fix by filling dims)
    expect(throughputState(model({ num_hidden_layers: 52, num_kv_layers: 6, hidden_size: null }))).toBe(
      "unsupported-arch",
    );
  });

  it("isThroughputModeled mirrors the 'modeled' state", () => {
    expect(isThroughputModeled(model())).toBe(true);
    expect(isThroughputModeled(model({ attention_type: "MSA", kv_elems_per_token: 1 }))).toBe(false);
  });

  it("num_kv_layers == num_hidden_layers is NOT a hybrid", () => {
    expect(isLinearAttentionHybrid(model({ num_hidden_layers: 60, num_kv_layers: 60 }))).toBe(false);
  });
});

describe("throughputState — against the live model list", () => {
  const models = modelsData as Model[];

  it("classifies every listed model into a valid state (no crashes)", () => {
    for (const m of models) {
      expect(["modeled", "unsupported-arch", "data-incomplete"]).toContain(throughputState(m));
    }
  });

  it("every model with a partial-KV layer count is unsupported-arch", () => {
    for (const m of models) {
      if (isLinearAttentionHybrid(m)) {
        expect(throughputState(m)).toBe("unsupported-arch");
      }
    }
  });

  it("streams/cost stay available regardless — throughput gating never excludes a model", () => {
    // The predicate only labels throughput; it must classify, never drop.
    expect(models.length).toBeGreaterThan(0);
    expect(models.every((m) => throughputState(m) != null)).toBe(true);
  });
});
