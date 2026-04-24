import { describe, it, expect } from "vitest";
import type { ApiPricingEntry, Model, GpuOffering, PresetGpuConfig } from "@/types";
import {
  computeAvgCostPerRequest,
  computeSelfHostingMonthlyCost,
  findGpuOfferingForConfig,
  computeSelfHostingCostForConfig,
  selfHostingStepCost,
  type CostConfig,
} from "../api-hosting-cost";
import { DEFAULT_ADVANCED_SETTINGS } from "../matrix-calculator";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const BASE_CONFIG: CostConfig = {
  requestsPerConversation: 10,
  cacheHitRate: 0.9,
  cacheTtlMin: 5,
  avgInputTokens: 1000,
  avgOutputTokens: 500,
};

function makeEntry(overrides: Partial<ApiPricingEntry>): ApiPricingEntry {
  return {
    model_name: "test-model",
    lab: "test",
    litellm_id: "test",
    input_cost_per_token: 0.000010,
    output_cost_per_token: 0.000030,
    cache_creation_input_token_cost: null,
    cache_read_input_token_cost: null,
    context_window: 200000,
    max_output_tokens: 4096,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// hasCaching — now driven solely by cache_read_input_token_cost
// ---------------------------------------------------------------------------

describe("computeAvgCostPerRequest — caching behavior", () => {
  it("applies cache discount when only cache_read is set (OpenAI/Google style)", () => {
    const entry = makeEntry({ cache_read_input_token_cost: 0.0000025 }); // cache_creation stays null
    const noCacheConfig = { ...BASE_CONFIG, cacheHitRate: 0 };
    const highCacheConfig = { ...BASE_CONFIG, cacheHitRate: 0.99, cacheTtlMin: 60 };

    const costNoCache = computeAvgCostPerRequest(entry, noCacheConfig);
    const costHighCache = computeAvgCostPerRequest(entry, highCacheConfig);

    expect(costHighCache).toBeLessThan(costNoCache);
  });

  it("applies cache discount when both cache costs are set (Anthropic style)", () => {
    const entry = makeEntry({
      cache_creation_input_token_cost: 0.0000125,
      cache_read_input_token_cost: 0.0000005,
    });
    const noCacheConfig = { ...BASE_CONFIG, cacheHitRate: 0 };
    const highCacheConfig = { ...BASE_CONFIG, cacheHitRate: 0.99, cacheTtlMin: 60 };

    const costNoCache = computeAvgCostPerRequest(entry, noCacheConfig);
    const costHighCache = computeAvgCostPerRequest(entry, highCacheConfig);

    expect(costHighCache).toBeLessThan(costNoCache);
  });

  it("is unaffected by cacheHitRate when cache_read is null (Google pre-caching)", () => {
    const entry = makeEntry({
      cache_creation_input_token_cost: null,
      cache_read_input_token_cost: null,
    });
    const lowCache = computeAvgCostPerRequest(entry, { ...BASE_CONFIG, cacheHitRate: 0.5 });
    const highCache = computeAvgCostPerRequest(entry, { ...BASE_CONFIG, cacheHitRate: 0.99 });

    expect(lowCache).toBeCloseTo(highCache, 10);
  });

  it("returns a positive cost for all entry types", () => {
    const entries = [
      makeEntry({ cache_read_input_token_cost: null }), // no caching
      makeEntry({ cache_read_input_token_cost: 0.000001 }), // partial caching
      makeEntry({ cache_creation_input_token_cost: 0.00002, cache_read_input_token_cost: 0.000001 }), // full caching
    ];
    for (const entry of entries) {
      expect(computeAvgCostPerRequest(entry, BASE_CONFIG)).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// Request-1 cache expiry behavior
// ---------------------------------------------------------------------------

describe("computeAvgCostPerRequest — request-1 cold-start", () => {
  it("suppresses cache on request 1 when TTL < inter-conversation gap (Anthropic 5-min TTL)", () => {
    const entry = makeEntry({
      cache_creation_input_token_cost: 0.0000125,
      cache_read_input_token_cost: 0.0000005,
    });
    // TTL=5 min < 60-min gap → request 1 is cold every time
    const shortTtlConfig = { ...BASE_CONFIG, cacheTtlMin: 5, requestsPerConversation: 1 };
    // TTL=null (provider with no TTL data) → also treated as expired
    const nullTtlConfig = { ...BASE_CONFIG, cacheTtlMin: null, requestsPerConversation: 1 };

    const costShortTtl = computeAvgCostPerRequest(entry, shortTtlConfig);
    const costNullTtl = computeAvgCostPerRequest(entry, nullTtlConfig);

    // At 1 request both should equal full input cost (no cache read savings)
    expect(costShortTtl).toBeCloseTo(costNullTtl, 10);
  });

  it("allows cache on request 1 when TTL >= inter-conversation gap (OpenAI 60-min TTL)", () => {
    const entry = makeEntry({ cache_read_input_token_cost: 0.000001 });
    // cacheTtlMin=60 is NOT < INTER_CONVERSATION_GAP_MIN(60) so request 1 gets cache
    const config60 = { ...BASE_CONFIG, cacheTtlMin: 60, requestsPerConversation: 1 };
    // cacheTtlMin=5 suppresses request-1 cache
    const config5 = { ...BASE_CONFIG, cacheTtlMin: 5, requestsPerConversation: 1 };

    const costWith60 = computeAvgCostPerRequest(entry, config60);
    const costWith5 = computeAvgCostPerRequest(entry, config5);

    expect(costWith60).toBeLessThan(costWith5);
  });

  it("lowers avg cost per request when context compaction kicks in more often at higher T", () => {
    // context_window=75000 with O=500 triggers compaction at request ~112.
    // Compacted requests reset ctx to COMPACTION_THRESHOLD (cheap), so T=300 has
    // proportionally more cheap compacted requests than T=150 → lower average.
    const entry = makeEntry({
      cache_creation_input_token_cost: 0.0000125,
      cache_read_input_token_cost: 0.0000005,
      context_window: 75_000,
    });
    const config150 = { ...BASE_CONFIG, cacheTtlMin: 5, requestsPerConversation: 150 };
    const config300 = { ...BASE_CONFIG, cacheTtlMin: 5, requestsPerConversation: 300 };

    const avg150 = computeAvgCostPerRequest(entry, config150);
    const avg300 = computeAvgCostPerRequest(entry, config300);

    expect(avg300).toBeLessThan(avg150);
  });
});

// ---------------------------------------------------------------------------
// Context-window compaction
// ---------------------------------------------------------------------------

describe("computeAvgCostPerRequest — context compaction", () => {
  it("resets context when it exceeds the context window", () => {
    const tinyContextEntry = makeEntry({ context_window: 21000 }); // just above COMPACTION_THRESHOLD
    const bigContextEntry = makeEntry({ context_window: 200000 });
    const config = { ...BASE_CONFIG, cacheTtlMin: null, cacheHitRate: 0, requestsPerConversation: 100 };

    const costTiny = computeAvgCostPerRequest(tinyContextEntry, config);
    const costBig = computeAvgCostPerRequest(bigContextEntry, config);

    // With tiny context, compaction fires often and keeps avg input small
    expect(costTiny).toBeLessThan(costBig);
  });
});

// ---------------------------------------------------------------------------
// computeSelfHostingMonthlyCost — scaled GPU fallback
// ---------------------------------------------------------------------------

// Kimi K-2.5: 1026.4B MoE INT4, needs ~683 GB with overhead.
// Only a x1 B300 (288 GB) is available — no real SKU fits, but 3× scaled does.
const KIMI_K25: Model = {
  model_name: "Kimi-K2.5",
  learnable_params_b: 1026.4,
  active_params_b: 32.9,
  architecture: "MoE",
  context_length: 262144,
  precision: "INT4",
  routed_expert_params_b: 1014.7,
  attention_type: "MLA",
  num_hidden_layers: 61,
  num_kv_layers: null,
  num_kv_heads: null,
  head_dim: null,
  kv_lora_rank: 512,
  qk_rope_head_dim: 64,
  hf_model_id: null,
  model_url: null,
};

const B300_X1: GpuOffering = {
  gpu_name: "B300",
  vram_gb: 288,
  gpu_count: 1,
  total_vram_gb: 288,
  price_per_hour: 6.09,
  currency: "EUR",
  provider: "substrate",
  instance_name: "substrate-B300-1x",
  location: "Europe",
  interconnect: "NVLink",
};

describe("computeSelfHostingMonthlyCost — scaled GPU fallback", () => {
  it("returns a cost for Kimi-K2.5 when only a x1 B300 is available (scaled to 3×)", () => {
    const cost = computeSelfHostingMonthlyCost(KIMI_K25, [B300_X1], DEFAULT_ADVANCED_SETTINGS);
    expect(cost).not.toBeNull();
    expect(cost!).toBeGreaterThan(0);
  });

  it("returns null when no GPU can fit the model even after scaling", () => {
    const tinyGpu: GpuOffering = { ...B300_X1, gpu_name: "A10", vram_gb: 24, total_vram_gb: 24 };
    const cost = computeSelfHostingMonthlyCost(KIMI_K25, [tinyGpu], DEFAULT_ADVANCED_SETTINGS);
    expect(cost).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// findGpuOfferingForConfig
// ---------------------------------------------------------------------------

const B300_CONFIG: PresetGpuConfig = {
  label: "B300 ×1",
  gpuName: "B300",
  gpuCount: 1,
  vramPerGpu: 288,
  totalVramGb: 288,
  interconnect: "NVLink",
};

const B300_X1_CHEAP: GpuOffering = { ...B300_X1, price_per_hour: 5.00, provider: "cheap-provider" };

describe("findGpuOfferingForConfig", () => {
  it("returns the matching offering when one exists", () => {
    const result = findGpuOfferingForConfig(B300_CONFIG, [B300_X1]);
    expect(result).toBe(B300_X1);
  });

  it("returns the cheapest matching offering when multiple exist", () => {
    const result = findGpuOfferingForConfig(B300_CONFIG, [B300_X1, B300_X1_CHEAP]);
    expect(result).toBe(B300_X1_CHEAP);
  });

  it("returns null when no offering matches the config", () => {
    const a10Config: PresetGpuConfig = { ...B300_CONFIG, gpuName: "A10", vramPerGpu: 24, totalVramGb: 24 };
    const result = findGpuOfferingForConfig(a10Config, [B300_X1]);
    expect(result).toBeNull();
  });

  it("returns null for an empty GPU list", () => {
    expect(findGpuOfferingForConfig(B300_CONFIG, [])).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// computeSelfHostingCostForConfig
// ---------------------------------------------------------------------------

const SMALL_MODEL: Model = {
  model_name: "Small-7B",
  learnable_params_b: 7,
  active_params_b: 7,
  architecture: "Dense",
  context_length: 32768,
  precision: "BF16",
  routed_expert_params_b: null,
  attention_type: "GQA",
  num_hidden_layers: 32,
  num_kv_layers: null,
  num_kv_heads: 8,
  head_dim: 128,
  kv_lora_rank: null,
  qk_rope_head_dim: null,
  hf_model_id: null,
  model_url: null,
};

describe("computeSelfHostingCostForConfig", () => {
  it("returns null when no offering matches the config", () => {
    const a10Config: PresetGpuConfig = { ...B300_CONFIG, gpuName: "A10", vramPerGpu: 24, totalVramGb: 24 };
    const result = computeSelfHostingCostForConfig(SMALL_MODEL, a10Config, [B300_X1], DEFAULT_ADVANCED_SETTINGS, 0.9);
    expect(result).toBeNull();
  });

  it("returns baseMonthlyCost = price_per_hour * 720 when offering matches", () => {
    const result = computeSelfHostingCostForConfig(SMALL_MODEL, B300_CONFIG, [B300_X1], DEFAULT_ADVANCED_SETTINGS, 0.9);
    expect(result).not.toBeNull();
    expect(result!.baseMonthlyCost).toBeCloseTo(B300_X1.price_per_hour * 720, 5);
  });

  it("returns positive maxRequestsPerMonth for a model that fits", () => {
    const result = computeSelfHostingCostForConfig(SMALL_MODEL, B300_CONFIG, [B300_X1], DEFAULT_ADVANCED_SETTINGS, 0.9);
    expect(result).not.toBeNull();
    expect(result!.maxRequestsPerMonth).not.toBeNull();
    expect(result!.maxRequestsPerMonth!).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// selfHostingStepCost
// ---------------------------------------------------------------------------

describe("selfHostingStepCost", () => {
  it("returns baseMonthlyCost when maxRequestsPerMonth is null", () => {
    const config = { baseMonthlyCost: 1000, maxRequestsPerMonth: null };
    expect(selfHostingStepCost(0, config)).toBe(1000);
    expect(selfHostingStepCost(999999, config)).toBe(1000);
  });

  it("returns baseMonthlyCost for x=0 (at least 1 replica)", () => {
    const config = { baseMonthlyCost: 500, maxRequestsPerMonth: 10000 };
    expect(selfHostingStepCost(0, config)).toBe(500);
  });

  it("returns 1× baseMonthlyCost for x within first tier", () => {
    const config = { baseMonthlyCost: 500, maxRequestsPerMonth: 10000 };
    expect(selfHostingStepCost(1, config)).toBe(500);
    expect(selfHostingStepCost(10000, config)).toBe(500);
  });

  it("returns 2× baseMonthlyCost for x just above one tier", () => {
    const config = { baseMonthlyCost: 500, maxRequestsPerMonth: 10000 };
    expect(selfHostingStepCost(10001, config)).toBe(1000);
    expect(selfHostingStepCost(20000, config)).toBe(1000);
  });

  it("scales linearly with replica count", () => {
    const config = { baseMonthlyCost: 500, maxRequestsPerMonth: 10000 };
    expect(selfHostingStepCost(20001, config)).toBe(1500);
    expect(selfHostingStepCost(30000, config)).toBe(1500);
  });
});
