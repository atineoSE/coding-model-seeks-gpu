import { describe, it, expect } from "vitest";
import type { ApiPricingEntry, Model, GpuOffering } from "@/types";
import { computeAvgCostPerTurn, computeSelfHostingMonthlyCost, type CostConfig } from "../api-hosting-cost";
import { DEFAULT_ADVANCED_SETTINGS } from "../matrix-calculator";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const BASE_CONFIG: CostConfig = {
  turnsPerConversation: 10,
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

describe("computeAvgCostPerTurn — caching behaviour", () => {
  it("applies cache discount when only cache_read is set (OpenAI/Google style)", () => {
    const entry = makeEntry({ cache_read_input_token_cost: 0.0000025 }); // cache_creation stays null
    const noCacheConfig = { ...BASE_CONFIG, cacheHitRate: 0 };
    const highCacheConfig = { ...BASE_CONFIG, cacheHitRate: 0.99, cacheTtlMin: 60 };

    const costNoCache = computeAvgCostPerTurn(entry, noCacheConfig);
    const costHighCache = computeAvgCostPerTurn(entry, highCacheConfig);

    expect(costHighCache).toBeLessThan(costNoCache);
  });

  it("applies cache discount when both cache costs are set (Anthropic style)", () => {
    const entry = makeEntry({
      cache_creation_input_token_cost: 0.0000125,
      cache_read_input_token_cost: 0.0000005,
    });
    const noCacheConfig = { ...BASE_CONFIG, cacheHitRate: 0 };
    const highCacheConfig = { ...BASE_CONFIG, cacheHitRate: 0.99, cacheTtlMin: 60 };

    const costNoCache = computeAvgCostPerTurn(entry, noCacheConfig);
    const costHighCache = computeAvgCostPerTurn(entry, highCacheConfig);

    expect(costHighCache).toBeLessThan(costNoCache);
  });

  it("is unaffected by cacheHitRate when cache_read is null (Google pre-caching)", () => {
    const entry = makeEntry({
      cache_creation_input_token_cost: null,
      cache_read_input_token_cost: null,
    });
    const lowCache = computeAvgCostPerTurn(entry, { ...BASE_CONFIG, cacheHitRate: 0.5 });
    const highCache = computeAvgCostPerTurn(entry, { ...BASE_CONFIG, cacheHitRate: 0.99 });

    expect(lowCache).toBeCloseTo(highCache, 10);
  });

  it("returns a positive cost for all entry types", () => {
    const entries = [
      makeEntry({ cache_read_input_token_cost: null }), // no caching
      makeEntry({ cache_read_input_token_cost: 0.000001 }), // partial caching
      makeEntry({ cache_creation_input_token_cost: 0.00002, cache_read_input_token_cost: 0.000001 }), // full caching
    ];
    for (const entry of entries) {
      expect(computeAvgCostPerTurn(entry, BASE_CONFIG)).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// Turn-1 cache expiry behaviour
// ---------------------------------------------------------------------------

describe("computeAvgCostPerTurn — turn-1 cold-start", () => {
  it("suppresses cache on turn 1 when TTL < inter-conversation gap (Anthropic 5-min TTL)", () => {
    const entry = makeEntry({
      cache_creation_input_token_cost: 0.0000125,
      cache_read_input_token_cost: 0.0000005,
    });
    // TTL=5 min < 60-min gap → turn 1 is cold every time
    const shortTtlConfig = { ...BASE_CONFIG, cacheTtlMin: 5, turnsPerConversation: 1 };
    // TTL=null (provider with no TTL data) → also treated as expired
    const nullTtlConfig = { ...BASE_CONFIG, cacheTtlMin: null, turnsPerConversation: 1 };

    const costShortTtl = computeAvgCostPerTurn(entry, shortTtlConfig);
    const costNullTtl = computeAvgCostPerTurn(entry, nullTtlConfig);

    // At 1 turn both should equal full input cost (no cache read savings)
    expect(costShortTtl).toBeCloseTo(costNullTtl, 10);
  });

  it("allows cache on turn 1 when TTL >= inter-conversation gap (OpenAI 60-min TTL)", () => {
    const entry = makeEntry({ cache_read_input_token_cost: 0.000001 });
    // cacheTtlMin=60 is NOT < INTER_CONVERSATION_GAP_MIN(60) so turn 1 gets cache
    const config60 = { ...BASE_CONFIG, cacheTtlMin: 60, turnsPerConversation: 1 };
    // cacheTtlMin=5 suppresses turn-1 cache
    const config5 = { ...BASE_CONFIG, cacheTtlMin: 5, turnsPerConversation: 1 };

    const costWith60 = computeAvgCostPerTurn(entry, config60);
    const costWith5 = computeAvgCostPerTurn(entry, config5);

    expect(costWith60).toBeLessThan(costWith5);
  });

  it("lowers avg cost per turn when context compaction kicks in more often at higher T", () => {
    // context_window=75000 with O=500 triggers compaction at turn ~112.
    // Compacted turns reset ctx to COMPACTION_THRESHOLD (cheap), so T=300 has
    // proportionally more cheap compacted turns than T=150 → lower average.
    const entry = makeEntry({
      cache_creation_input_token_cost: 0.0000125,
      cache_read_input_token_cost: 0.0000005,
      context_window: 75_000,
    });
    const config150 = { ...BASE_CONFIG, cacheTtlMin: 5, turnsPerConversation: 150 };
    const config300 = { ...BASE_CONFIG, cacheTtlMin: 5, turnsPerConversation: 300 };

    const avg150 = computeAvgCostPerTurn(entry, config150);
    const avg300 = computeAvgCostPerTurn(entry, config300);

    expect(avg300).toBeLessThan(avg150);
  });
});

// ---------------------------------------------------------------------------
// Context-window compaction
// ---------------------------------------------------------------------------

describe("computeAvgCostPerTurn — context compaction", () => {
  it("resets context when it exceeds the context window", () => {
    const tinyContextEntry = makeEntry({ context_window: 21000 }); // just above COMPACTION_THRESHOLD
    const bigContextEntry = makeEntry({ context_window: 200000 });
    const config = { ...BASE_CONFIG, cacheTtlMin: null, cacheHitRate: 0, turnsPerConversation: 100 };

    const costTiny = computeAvgCostPerTurn(tinyContextEntry, config);
    const costBig = computeAvgCostPerTurn(bigContextEntry, config);

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
