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
  attention_type: "MLA" | "GQA" | null;
  num_hidden_layers: number | null;
  num_kv_heads: number | null;
  head_dim: number | null;
  kv_lora_rank: number | null;
  qk_rope_head_dim: number | null;
  hf_model_id: string | null;
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
}

export interface MatrixCell {
  model: Model;
  benchmark: BenchmarkScore;
  sotaScore: SotaScore | null;
  percentOfSota: number;
  gpuSetups: GpuSetupOption[];
  costPerStreamPerMonth: number | null;
  exceedsCapacity: boolean;
  decodeThroughputTokS: number | null;
  utilization: number | null;
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

export interface AdvancedSettings {
  avgInputTokens: number;
  avgOutputTokens: number;
  minTokPerStream: number;
  prefixCacheHitRate: number;
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

