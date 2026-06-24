/**
 * Single-stream decode LATENCY model (first-principles, public inputs only).
 *
 * Unlike a pure bandwidth roofline (`tok/s = hbm_bw / active_bytes`), this model
 * accounts for the serial critical path of a single in-flight decode step:
 *
 *   per_token_time =
 *       weight-read         active_bytes_per_gpu / hbm_bw
 *     + TP all-reduce       layers × (2·hidden·bytes / interconnect_bw + collective_latency)
 *     + EP all-to-all       layers × (top-k·hidden·bytes / interconnect_bw + collective_latency)   (MoE only)
 *     + kernel launches     launch_latency × layers
 *
 *   singleStreamTokS = 1 / per_token_time
 *
 * Every term is derived from public physical inputs: GPU datasheet bandwidths,
 * the interconnect tier, model architecture dimensions, and generic
 * publicly-citable hardware constants (NCCL collective latency, CUDA kernel
 * launch latency). There are NO fitted or privately-measured constants here.
 *
 * Because the comm and launch terms are strictly positive, the result is always
 * strictly below the pure bandwidth roofline (see {@link bandwidthRooflineTokS}).
 */

import type { InterconnectTier } from "@/types";

/** Bytes per binary GB / TB, matching the convention in `calculations.ts`. */
const BYTES_PER_GB = 1024 ** 3;
const BYTES_PER_TB = 1024 ** 4;

/**
 * Decode activations stay in bf16/fp16 (2 bytes/elem) on the wire regardless of
 * how the weights are quantized — quantized weights are dequantized to the
 * compute dtype before the matmul, and the all-reduce/all-to-all payloads are
 * the bf16/fp16 activations.
 */
export const DEFAULT_ACTIVATION_BYTES = 2;

/**
 * CUDA kernel launch latency, ~5 µs on modern GPUs. This is a generic, public
 * hardware figure (NVIDIA CUDA C++ Programming Guide; widely reproduced in
 * launch-overhead microbenchmarks). Each transformer layer issues at least one
 * launch on the critical path; we charge one launch per layer.
 */
export const DEFAULT_KERNEL_LAUNCH_LATENCY_S = 5e-6;

/**
 * Per-collective fixed latency by interconnect tier (seconds).
 *
 * NCCL small-message collective latency is a few microseconds intra-node over
 * NVLink/NVSwitch and higher over PCIe, as published by NVIDIA's nccl-tests
 * (`all_reduce_perf`, see https://github.com/NVIDIA/nccl-tests and the NVIDIA
 * NCCL documentation). These are generic, publicly-citable hardware constants,
 * not measurements from any private benchmark.
 */
export const DEFAULT_COLLECTIVE_LATENCY_S: Record<InterconnectTier, number> = {
  nvswitch: 5e-6,
  nvlink_paired: 8e-6,
  none: 15e-6,
};

/** GPU physical inputs for the decode-latency model. */
export interface DecodeLatencyGpu {
  /** HBM memory bandwidth in TB/s (datasheet). */
  hbmBandwidthTbS: number;
  /** Inter-GPU interconnect tier. */
  interconnectTier: InterconnectTier;
  /**
   * Effective unidirectional inter-GPU link bandwidth in GB/s (NVLink/NVSwitch
   * for `nvswitch`/`nvlink_paired`, PCIe link bandwidth for `none`).
   */
  interconnectBandwidthGbS: number;
  /**
   * Per-collective fixed latency in seconds. Defaults to a tier-appropriate
   * public NCCL figure (see {@link DEFAULT_COLLECTIVE_LATENCY_S}).
   */
  collectiveLatencyS?: number;
}

/** Model architecture inputs for the decode-latency model. */
export interface DecodeLatencyModelDims {
  /** Number of transformer layers. */
  numLayers: number;
  /** Residual-stream width (HF config `hidden_size`). */
  hiddenSize: number;
  /** Active parameters in billions read from HBM per token. */
  activeParamsB: number;
  /** Bytes per parameter at the serving precision. */
  bytesPerParam: number;
  /** Whether the model is MoE (routes tokens through an all-to-all). */
  isMoe: boolean;
  /** Top-k experts activated per token (MoE only). */
  topK: number | null;
  /** Total routed experts (MoE only); accepted as part of the public dims. */
  numExperts?: number | null;
  /** MLA latent rank `kv_lora_rank` (accepted as part of the public dims). */
  kvLoraRank?: number | null;
  /** MLA RoPE head dim `qk_rope_head_dim` (accepted as part of the public dims). */
  qkRopeHeadDim?: number | null;
  /** Activation byte width on the wire. Defaults to bf16/fp16 (2 bytes). */
  activationBytes?: number;
}

/** Parallelism / runtime inputs for the decode-latency model. */
export interface DecodeLatencyParams {
  /** Tensor-parallel degree (and, for MoE, the expert-parallel degree). */
  tp: number;
  /** CUDA kernel launch latency in seconds. Defaults to a public figure. */
  launchLatencyS?: number;
}

/** Per-token critical-path breakdown plus the resulting single-stream tok/s. */
export interface DecodeLatencyResult {
  /** Total per-token time in seconds. */
  perTokenS: number;
  /** Single-stream decode throughput in tok/s = 1 / perTokenS. */
  singleStreamTokS: number;
  /** Pure bandwidth-roofline tok/s (weight-read only); always ≥ singleStreamTokS. */
  bandwidthRooflineTokS: number;
  /** Critical-path contribution of each term, in seconds. */
  breakdown: {
    weightReadS: number;
    tpAllReduceS: number;
    epAllToAllS: number;
    launchS: number;
  };
}

/**
 * Pure bandwidth roofline for a single decode stream (tok/s).
 *
 * This is the weight-read term in isolation: every active byte is sharded
 * across `tp` GPUs and streamed from HBM once per token. The latency model adds
 * communication and launch overhead on top, so it always lands strictly below
 * this ceiling.
 */
export function bandwidthRooflineTokS(gpu: DecodeLatencyGpu, model: DecodeLatencyModelDims, tp: number): number {
  const hbmBytesPerS = gpu.hbmBandwidthTbS * BYTES_PER_TB;
  const activeBytes = model.activeParamsB * model.bytesPerParam * BYTES_PER_GB;
  const activeBytesPerGpu = activeBytes / Math.max(1, tp);
  const weightReadS = activeBytesPerGpu / hbmBytesPerS;
  return 1 / weightReadS;
}

/**
 * Compute the single-stream decode latency model.
 *
 * @see module docstring for the per-token formula.
 */
export function calcDecodeLatency(
  gpu: DecodeLatencyGpu,
  model: DecodeLatencyModelDims,
  params: DecodeLatencyParams,
): DecodeLatencyResult {
  const tp = Math.max(1, params.tp);

  const hbmBytesPerS = gpu.hbmBandwidthTbS * BYTES_PER_TB;
  const interconnectBytesPerS = gpu.interconnectBandwidthGbS * BYTES_PER_GB;

  const collectiveLatencyS = gpu.collectiveLatencyS ?? DEFAULT_COLLECTIVE_LATENCY_S[gpu.interconnectTier];
  const launchLatencyS = params.launchLatencyS ?? DEFAULT_KERNEL_LAUNCH_LATENCY_S;
  const activationBytes = model.activationBytes ?? DEFAULT_ACTIVATION_BYTES;

  // 1. Weight read: active weights sharded across tp GPUs, streamed once/token.
  const activeBytes = model.activeParamsB * model.bytesPerParam * BYTES_PER_GB;
  const activeBytesPerGpu = activeBytes / tp;
  const weightReadS = activeBytesPerGpu / hbmBytesPerS;

  // 2. TP all-reduce: one all-reduce of the hidden activation per layer. The
  // factor 2 is the ring all-reduce data-volume factor (~2·(N−1)/N ≈ 2).
  let tpAllReduceS = 0;
  if (tp > 1) {
    const allReduceBytes = 2 * model.hiddenSize * activationBytes;
    tpAllReduceS = model.numLayers * (allReduceBytes / interconnectBytesPerS + collectiveLatencyS);
  }

  // 3. EP all-to-all (MoE only): each token's hidden vector is dispatched to its
  // top-k experts (and combined back), once per layer.
  let epAllToAllS = 0;
  if (model.isMoe && tp > 1 && model.topK !== null && (model.topK ?? 0) > 0) {
    const dispatchBytes = model.topK * model.hiddenSize * activationBytes;
    epAllToAllS = model.numLayers * (dispatchBytes / interconnectBytesPerS + collectiveLatencyS);
  }

  // 4. Kernel launches: one launch on the critical path per layer.
  const launchS = launchLatencyS * model.numLayers;

  const perTokenS = weightReadS + tpAllReduceS + epAllToAllS + launchS;

  return {
    perTokenS,
    singleStreamTokS: 1 / perTokenS,
    bandwidthRooflineTokS: 1 / weightReadS,
    breakdown: { weightReadS, tpAllReduceS, epAllToAllS, launchS },
  };
}
