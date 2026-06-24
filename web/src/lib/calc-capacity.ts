/**
 * First-principles VRAM capacity model.
 *
 * Two pure functions:
 *  - `calcRuntimeReserve` — the non-KV runtime VRAM an inference engine pins
 *    on top of the model weights (activation buffers, captured CUDA graphs,
 *    and TP/EP collective staging buffers).
 *  - `calcOperatingStreams` — how many concurrent streams the leftover VRAM
 *    (usable − weights − reserve) can admit, returned as a low/high band over
 *    a plausible prefix-reuse range.
 *
 * Every term is derived from PUBLIC inputs only: model architecture dims
 * (hidden size, layers, top-k experts), the parallelism layout, and published
 * engine defaults (exposed as overridable inputs, never fitted to a private
 * benchmark). The KV-per-token width is reused read-only from calculations.ts.
 */

import type { Model } from "@/types";
import { calcKvCachePerToken } from "./calculations";

const BYTES_PER_GB = 1024 * 1024 * 1024;

/**
 * Parallelism layout the deployment runs under. Produced by the topology model
 * (sibling module); only the degrees relevant to runtime memory are used here.
 */
export interface ParallelLayout {
  /** Total GPUs in the deployment (the runtime reserve is paid on each). */
  numGpus: number;
  /** Tensor-parallel degree. tp > 1 ⇒ an all-reduce staging buffer is pinned. */
  tp: number;
  /** Expert-parallel degree (MoE). ep > 1 ⇒ an all-to-all dispatch buffer is pinned. */
  ep: number;
  /** Pipeline-parallel degree. Splits the layers across stages; default 1. */
  pp?: number;
}

/**
 * Public engine defaults that size the runtime reserve. These are documented
 * vLLM defaults, exposed here so they can be overridden per deployment rather
 * than baked in as magic constants.
 */
export interface EngineDefaults {
  /**
   * Max tokens batched into a single forward pass (vLLM
   * `--max-num-batched-tokens`; V1 default 8192). Drives the activation-peak
   * and collective-buffer widths.
   * https://docs.vllm.ai/en/latest/serving/engine_args.html
   */
  maxBatchedTokens: number;
  /**
   * Bytes per activation element. Activations stay in the bf16/fp16 compute
   * dtype (2 bytes) even when the weights are quantized.
   */
  bytesPerActivation: 2 | 4;
  /**
   * Number of residual-stream-sized activation tensors live at the peak of a
   * forward pass. A transformer block holds, at minimum, its input residual
   * plus the largest concurrently-materialized intermediate ⇒ 2 (conservative,
   * intermediate widths are not a public field of every model).
   */
  activationBuffers: number;
  /**
   * Largest decode batch captured into a CUDA graph. vLLM captures graphs for
   * batch sizes up to `max_num_seqs` (default 256); the captured graphs pin
   * per-layer decode scratch for that batch.
   * https://docs.vllm.ai/en/latest/serving/engine_args.html
   */
  cudaGraphMaxBatchSize: number;
}

/** Documented vLLM V1 defaults — overridable, not fitted to any measurement. */
export const DEFAULT_ENGINE_DEFAULTS: EngineDefaults = {
  maxBatchedTokens: 8192,
  bytesPerActivation: 2,
  activationBuffers: 2,
  cudaGraphMaxBatchSize: 256,
};

/**
 * A plausible operating range for the shared/cached prefix fraction. Centered
 * on the project's 0.5 default (see AdvancedSettings.prefixReuse): low reuse is
 * the conservative end (fewer streams fit), high reuse the optimistic end.
 */
export const DEFAULT_PREFIX_REUSE_RANGE = { low: 0.25, high: 0.75 };

/**
 * Estimate the non-KV runtime VRAM (GB, summed across the whole layout) an
 * engine pins on top of the model weights.
 *
 * Per-GPU terms, all from public dims (`hidden` = residual-stream width,
 * `bytes` = activation dtype, `T` = max batched tokens, `topK` = experts/token):
 *
 *  - Activation peak  = hidden × T × bytes × activationBuffers
 *      Residual-stream-sized working tensors held during a forward pass.
 *  - CUDA-graph scratch = layersPerGpu × hidden × bytes × cudaGraphMaxBatchSize
 *      Captured decode graphs pin per-layer scratch for the largest batch.
 *  - TP all-reduce buffer = hidden × T × bytes        (only when tp > 1)
 *      Staging buffer for the post-block all-reduce over the residual stream.
 *  - EP all-to-all buffer = hidden × T × topK × bytes (only when ep > 1, MoE)
 *      Each token's hidden state is dispatched to its top-k experts.
 *
 * The reserve is paid on every GPU, so the per-GPU sum is multiplied by
 * `numGpus`. Layers are split across pipeline stages (layersPerGpu = layers/pp)
 * so the CUDA-graph scratch is not double-counted under pipeline parallelism.
 *
 * Returns 0 when the required public dims are missing.
 */
export function calcRuntimeReserve(
  model: Model,
  layout: ParallelLayout,
  engineDefaults: EngineDefaults = DEFAULT_ENGINE_DEFAULTS,
): number {
  const hidden = model.hidden_size;
  const layers = model.num_hidden_layers;
  if (hidden === null || layers === null) return 0;
  if (layout.numGpus <= 0) return 0;

  const { maxBatchedTokens: T, bytesPerActivation: bytes } = engineDefaults;
  const pp = layout.pp ?? 1;
  const layersPerGpu = layers / Math.max(1, pp);
  const isMoe = model.architecture === "MoE";
  const topK = isMoe ? model.experts_per_token ?? 1 : 1;

  // Per-GPU bytes.
  const activationBytes = hidden * T * bytes * engineDefaults.activationBuffers;
  const cudaGraphBytes =
    layersPerGpu * hidden * bytes * engineDefaults.cudaGraphMaxBatchSize;
  const tpBufferBytes = layout.tp > 1 ? hidden * T * bytes : 0;
  const epBufferBytes = isMoe && layout.ep > 1 ? hidden * T * topK * bytes : 0;

  const perGpuBytes =
    activationBytes + cudaGraphBytes + tpBufferBytes + epBufferBytes;

  return (perGpuBytes * layout.numGpus) / BYTES_PER_GB;
}

export interface OperatingStreamsInput {
  model: Model;
  /** Usable VRAM (GB) after the engine's memory-utilization cap. */
  usableVramGb: number;
  /** Model weight footprint (GB) across the whole layout. */
  weightsGb: number;
  /** Non-KV runtime reserve (GB) — typically `calcRuntimeReserve(...)`. */
  reserveGb: number;
  avgInputTokens: number;
  avgOutputTokens: number;
  /** 1 for FP8 KV, 2 for FP16 KV (default 2). */
  kvPrecisionBytes?: 1 | 2;
  /** Plausible shared-prefix fraction range (0–1). */
  prefixReuseRange?: { low: number; high: number };
}

/**
 * Concurrent streams the leftover VRAM can admit, as a low/high band.
 *
 *   streams(p) = floor( (usable − weights − reserve)
 *                       / (contextTokens × kvPerToken × (1 − p)) )
 *
 * Whole-prompt admission: a stream reserves KV for its entire prompt
 * (input + output) up front, so the per-stream cost uses the full context.
 * Prefix reuse `p` is the fraction of that context shared across streams and
 * served from the prefix cache, so the *marginal* KV per stream is scaled by
 * (1 − p). Higher reuse ⇒ smaller per-stream cost ⇒ more streams (the `high`
 * end); lower reuse ⇒ the `low` end.
 *
 * Returns { low: 0, high: 0 } when no VRAM is left or the KV width is unknown.
 */
export function calcOperatingStreams(
  input: OperatingStreamsInput,
): { low: number; high: number } {
  const {
    model,
    usableVramGb,
    weightsGb,
    reserveGb,
    avgInputTokens,
    avgOutputTokens,
    kvPrecisionBytes = 2,
    prefixReuseRange = DEFAULT_PREFIX_REUSE_RANGE,
  } = input;

  const kvBudgetGb = usableVramGb - weightsGb - reserveGb;
  if (kvBudgetGb <= 0) return { low: 0, high: 0 };

  const kvPerTokenGb = calcKvCachePerToken(model, kvPrecisionBytes);
  if (kvPerTokenGb <= 0) return { low: 0, high: 0 };

  const contextTokens = avgInputTokens + avgOutputTokens;
  if (contextTokens <= 0) return { low: 0, high: 0 };

  const contextGb = contextTokens * kvPerTokenGb;

  // Clamp reuse to [0, 1); p = 1 would mean zero marginal cost (unbounded).
  const clamp = (p: number) => Math.min(0.999, Math.max(0, p));
  const streamsAt = (p: number): number => {
    const perStreamGb = contextGb * (1 - clamp(p));
    if (perStreamGb <= 0) return 0;
    return Math.max(0, Math.floor(kvBudgetGb / perStreamGb));
  };

  // Higher reuse ⇒ more streams; lower reuse ⇒ fewer streams.
  return {
    low: streamsAt(prefixReuseRange.low),
    high: streamsAt(prefixReuseRange.high),
  };
}
