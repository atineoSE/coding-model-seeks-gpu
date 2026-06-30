export interface GpuSource {
  service_name: string;
  service_url: string;
  description: string;
  currency: string;
  currency_symbol: string;
}

export interface GpuOffering {
  gpu_name: string;
  vram_gb: number;
  gpu_count: number;
  total_vram_gb: number;
  price_per_hour: number;
  currency: string;
  provider: string;
  instance_name: string;
  location: string;
  interconnect: string | null;
}

export interface Model {
  model_name: string;
  learnable_params_b: number | null;
  active_params_b: number | null;
  architecture: "Dense" | "MoE";
  context_length: number | null;
  precision: string | null;
  routed_expert_params_b: number | null;
  attention_type: "MLA" | "GQA" | "DSV4" | "MSA" | null;
  num_hidden_layers: number | null;
  /** Transformer residual-stream width (HF config `hidden_size`). */
  hidden_size: number | null;
  num_kv_layers: number | null;
  num_kv_heads: number | null;
  head_dim: number | null;
  /** Total routed experts for MoE models (HF config `n_routed_experts`); null for dense. */
  num_experts: number | null;
  /** Experts activated per token / top-k (HF config `num_experts_per_tok`); null for dense. */
  experts_per_token: number | null;
  // MLA latent dimensions (DeepSeek-style multi-head latent attention).
  kv_lora_rank: number | null;
  qk_rope_head_dim: number | null;
  kv_elems_per_token: number | null;
  hf_model_id: string | null;
  model_url: string | null;
  license_name: string | null;
  license_url: string | null;
}

export interface BenchmarkScore {
  model_name: string;
  benchmark_name: string;
  benchmark_display_name: string;
  score: number | null;
  rank: number | null;
  cost_per_task: number | null;
  benchmark_group: string;
  benchmark_group_display: string;
  openness?: "closed_api_available" | "open_weights";
}

export type Precision = "fp32" | "fp16" | "bf16" | "fp8" | "int8" | "int4";
export type PrecisionDisplay = "fp32" | "fp16/bf16" | "fp8" | "int8" | "int4";

// v0.2 Coding Model Seeks GPU types

export type Persona = "performance" | "budget" | "trends";

export type ConcurrencyTier = "single_agent" | "multi_agent" | "agent_fleet" | "agent_swarm";

export interface ConcurrencyTierConfig {
  key: ConcurrencyTier;
  label: string;
  minStreams: number;
  maxStreams: number;
  midpoint: number;
  description: string;
}

export interface SotaScore {
  benchmark_name: string;
  benchmark_display_name: string;
  sota_model_name: string;
  sota_score: number;
}

export interface GpuSetupOption {
  gpuName: string;
  gpuCount: number;
  interconnect: string | null;
  totalVramGb: number;
  monthlyCost: number;
  costPerStreamPerMonth: number;
  decodeThroughputTokS: number | null;
  maxConcurrentStreams: number;
  isProjected?: boolean;
  // First-principles throughput estimate for this (model, GPU layout), computed
  // in the matrix calculator. null when GPU specs / model dims / layout make a
  // first-principles estimate impossible. The result views render this
  // read-only; they never recompute it.
  deploymentEstimate: DeploymentEstimate | null;
}

export interface MatrixCell {
  model: Model;
  // null for unranked models (sized, no OpenHands Index score yet).
  benchmark: BenchmarkScore | null;
  sotaScore: SotaScore | null;
  // null for unranked models — never coerce a missing score to a number.
  percentOfSota: number | null;
  totalBenchmarkCost: number | null;
  gpuSetups: GpuSetupOption[];
  costPerStreamPerMonth: number | null;
  exceedsCapacity: boolean;
  decodeThroughputTokS: number | null;
  utilization: number | null;
  // True when the model has no benchmark score in this category (partial-model-data skill).
  isUnranked: boolean;
}

export interface PresetGpuConfig {
  label: string;
  gpuName: string;
  gpuCount: number;
  vramPerGpu: number;
  totalVramGb: number;
  interconnect: string | null;
}

export type KvCachePrecision = "auto" | "fp16" | "fp8";

/**
 * Normalized inter-GPU interconnect tier (derived from each GPU's NVLink/NVSwitch
 * form factor):
 * - "none"          → PCIe-only, no inter-GPU NVLink
 * - "nvlink_paired" → NVLink bridge connecting GPUs in 2-way pairs
 * - "nvswitch"      → SXM/superchip GPUs on an all-to-all NVSwitch fabric
 */
export type InterconnectTier = "none" | "nvlink_paired" | "nvswitch";

export interface AdvancedSettings {
  avgInputTokens: number;
  avgOutputTokens: number;
  /**
   * Fraction (0–1) of the input prompt that is shared/cached prefix across
   * streams, eligible for prefix-cache reuse. Default 0.5.
   */
  prefixReuse: number;
  minTokPerStream: number;
  /**
   * KV cache precision used for VRAM/concurrency budgeting.
   * - "fp16" → 2 bytes/elem (default; matches vLLM `kv_cache_dtype=auto`, which
   *   follows the model's bf16/fp16 compute dtype regardless of weight quant)
   * - "auto" → 1 byte/elem on FP8-KV-capable GPUs (Hopper+/Ada/Blackwell),
   *   falling back to 2 bytes elsewhere (opt-in FP8 KV cache)
   */
  kvCachePrecision: KvCachePrecision;
}

export interface CostResult {
  model: Model;
  benchmark: BenchmarkScore | null;
  memoryGb: number | null;
  gpusNeeded: number | null;
  gpuName: string | null;
  pricePerGpuHour: number | null;
  monthlyCost: number | null;
}

/**
 * First-principles deployment throughput estimate for a (model, GPU layout)
 * pairing. All figures are derived from public physical inputs — there are no
 * fitted or measured constants here.
 */
/**
 * Whether throughput (single-stream / aggregate tok/s) is modeled for a model.
 * Streams + cost are robust for every architecture; throughput is only
 * trustworthy for standard MoE/Dense + GQA/MLA. Mirrors `ThroughputState` in
 * `lib/throughput-support.ts`.
 */
export type ThroughputState = "modeled" | "unsupported-arch" | "data-incomplete";

export interface DeploymentEstimate {
  /**
   * Decode throughput (tok/s) for a single in-flight request. Null when
   * throughput is not modeled for this architecture (see `throughputState`).
   */
  singleStreamTokS: number | null;
  /** Concurrency the layout can sustain, as a low/high operating band. Always present. */
  operatingStreams: { low: number; high: number };
  /** Aggregate decode throughput (tok/s). Null when throughput is not modeled. */
  aggregateTokS: number | null;
  /** Whether the throughput figures above are modeled, and if not, why. */
  throughputState: ThroughputState;
  /** The public assumptions the estimate was derived under. */
  assumptions: {
    /** Context window used: average input + output tokens, and prefix reuse. */
    context: { avgInputTokens: number; avgOutputTokens: number; prefixReuse: number };
    /** Inter-GPU interconnect tier of the GPU layout. */
    interconnectTier: InterconnectTier;
    /** Whether throughput was modeled as MoE (active params) or dense. */
    moe: boolean;
  };
}

export interface ApiPricingEntry {
  model_name: string;
  lab: string;
  litellm_id: string;
  input_cost_per_token: number | null;
  output_cost_per_token: number | null;
  cache_creation_input_token_cost: number | null;
  cache_read_input_token_cost: number | null;
  context_window: number | null;
  max_output_tokens: number | null;
  // Additional LiteLLM fields preserved as passthrough
  [key: string]: unknown;
}

