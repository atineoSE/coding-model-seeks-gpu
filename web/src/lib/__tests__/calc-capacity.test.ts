import { describe, it, expect } from "vitest";
import type { Model } from "@/types";
import {
  calcRuntimeReserve,
  calcOperatingStreams,
  DEFAULT_ENGINE_DEFAULTS,
  type ParallelLayout,
} from "../calc-capacity";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// MoE model with public hidden/layer/top-k dims and a GQA KV width so both
// the runtime reserve and the KV-per-token cost are well-defined.
const MOE_MODEL: Model = {
  model_name: "Test-MoE",
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
  kv_elems_per_token: null,
  hf_model_id: null,
  model_url: null,
  license_name: null,
  license_url: null,
};

const MOE_EP_LAYOUT: ParallelLayout = { numGpus: 8, tp: 8, ep: 8, pp: 1 };

// ---------------------------------------------------------------------------
// calcRuntimeReserve
// ---------------------------------------------------------------------------

describe("calcRuntimeReserve", () => {
  it("is > 0 for an MoE + expert-parallel layout", () => {
    const reserve = calcRuntimeReserve(MOE_MODEL, MOE_EP_LAYOUT);
    expect(reserve).toBeGreaterThan(0);
  });

  it("grows when expert parallelism adds the all-to-all dispatch buffer", () => {
    const withEp = calcRuntimeReserve(MOE_MODEL, MOE_EP_LAYOUT);
    const withoutEp = calcRuntimeReserve(MOE_MODEL, {
      ...MOE_EP_LAYOUT,
      ep: 1,
    });
    expect(withEp).toBeGreaterThan(withoutEp);
  });

  it("returns 0 when required public dims are missing", () => {
    const noDims: Model = { ...MOE_MODEL, hidden_size: null };
    expect(calcRuntimeReserve(noDims, MOE_EP_LAYOUT)).toBe(0);
  });

  it("scales with the number of GPUs in the layout", () => {
    const oneGpu = calcRuntimeReserve(MOE_MODEL, {
      numGpus: 1,
      tp: 1,
      ep: 1,
    });
    const eightGpu = calcRuntimeReserve(MOE_MODEL, {
      numGpus: 8,
      tp: 1,
      ep: 1,
    });
    expect(eightGpu).toBeCloseTo(oneGpu * 8, 6);
  });
});

// ---------------------------------------------------------------------------
// calcOperatingStreams
// ---------------------------------------------------------------------------

describe("calcOperatingStreams", () => {
  const reserve = calcRuntimeReserve(MOE_MODEL, MOE_EP_LAYOUT, DEFAULT_ENGINE_DEFAULTS);

  const baseInput = {
    model: MOE_MODEL,
    usableVramGb: 8 * 141 * 0.9, // 8× 141GB GPUs at 90% utilization
    weightsGb: 470, // ~235B params at bf16
    reserveGb: reserve,
    avgInputTokens: 8000,
    avgOutputTokens: 1000,
  };

  it("returns a low/high band with high >= low", () => {
    const { low, high } = calcOperatingStreams(baseInput);
    expect(high).toBeGreaterThanOrEqual(low);
    expect(low).toBeGreaterThan(0);
  });

  it("shrinks as the context grows", () => {
    const small = calcOperatingStreams({ ...baseInput, avgInputTokens: 4000 });
    const large = calcOperatingStreams({ ...baseInput, avgInputTokens: 64000 });
    expect(large.low).toBeLessThan(small.low);
    expect(large.high).toBeLessThan(small.high);
  });

  it("returns zero streams when no VRAM is left after weights + reserve", () => {
    const { low, high } = calcOperatingStreams({
      ...baseInput,
      usableVramGb: baseInput.weightsGb + baseInput.reserveGb,
    });
    expect(low).toBe(0);
    expect(high).toBe(0);
  });

  it("admits more streams as prefix reuse rises", () => {
    const { low, high } = calcOperatingStreams({
      ...baseInput,
      prefixReuseRange: { low: 0.0, high: 0.9 },
    });
    expect(high).toBeGreaterThan(low);
  });
});
