/**
 * Single-stream decode LATENCY model (first-principles, public inputs only).
 *
 * Unlike a pure bandwidth roofline (`tok/s = hbm_bw / active_bytes`), this model
 * accounts for the serial critical path of a single in-flight decode step:
 *
 *   per_token_time =
 *       weight-read         active_bytes_per_gpu / hbm_bw
 *     + TP all-reduce       layers × (2·hidden·bytes / interconnect_bw + collective_latency)
 *     + EP all-to-all       layers × 2 × (top-k·hidden·bytes / interconnect_bw + collective_latency)   (MoE only;
 *                                          2 = dispatch + combine)
 *     + op-chain latency    launch_latency × layers × compute_ops_per_layer
 *     + PP send/recv        (pp − 1) × (hidden·bytes / interconnect_bw + collective_latency)
 *
 *   The weight-read also charges expert-parallel load imbalance: at batch-1 the
 *   busiest EP GPU reads E[max] full experts (balls-in-bins), not the balanced
 *   top-k/ep. The reported bandwidth roofline stays the *balanced* ceiling.
 *
 *   singleStreamTokS = 1 / per_token_time
 *
 * At batch-1 the dominant term is the op-chain latency: a decode step is a long
 * sequence of tiny, memory-latency-bound kernels that run serially with nothing
 * to overlap them, so per-op launch/dispatch latency — not HBM bandwidth — sets
 * the pace (measured MBU is single-digit %, far below the weight-read roofline).
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
 * Per-op serialization latency on the decode critical path, ~5 µs. A generic,
 * public hardware figure: the CUDA kernel launch / dispatch latency (NVIDIA CUDA
 * C++ Programming Guide; widely reproduced in launch-overhead microbenchmarks).
 * At batch-1 each dependent op on the critical path pays this once and cannot
 * overlap it with anything (there is no concurrent request to hide it behind).
 */
export const DEFAULT_KERNEL_LAUNCH_LATENCY_S = 5e-6;

/**
 * Number of serially-dependent COMPUTE ops on the critical path per layer
 * (collectives are excluded — they are charged separately by the TP all-reduce
 * and EP all-to-all terms). This is a structural count of the kernels a decoder
 * block issues in sequence, not a fitted constant:
 *
 *   dense block (~8): input RMSNorm → QKV proj → RoPE → attention → O proj →
 *     post-attn RMSNorm → gate+up proj → SiLU·mul → down proj
 *   MoE block (~11): the dense ops above + router/gating + the routed-expert
 *     gate/up/down GEMVs that the shared MLP does not have.
 *
 * At batch-1 every one of these is a tiny memory-latency-bound GEMV/kernel that
 * runs serially, so the per-op latency floor (not bandwidth) dominates. The
 * counts are order-of-magnitude structural estimates; expose them as overrides
 * rather than treating them as exact.
 */
export const DEFAULT_DECODE_COMPUTE_OPS_PER_LAYER = { dense: 8, moe: 11 } as const;

/**
 * Expected maximum bin load when `balls` are thrown uniformly into `bins` — the
 * classic balls-in-bins expected maximum, under the standard independence
 * approximation (treat each bin as Binomial(balls, 1/bins)):
 *
 *   E[max] = Σ_{m≥1} (1 − F(m−1)^bins),   F = Binomial(balls, 1/bins) CDF
 *
 * Used for expert-parallel load imbalance at batch-1: one token's `top_k` expert
 * GEMVs land on whichever EP GPUs host those experts, and the expert stage waits
 * for the busiest GPU. This is a *per-token* (per-decode-step) effect, so it is
 * NOT removed by the aggregate load-balancing that MoE training enforces. A
 * public combinatorial result, not a fitted constant.
 */
export function expectedMaxBinLoad(balls: number, bins: number): number {
  if (bins <= 1) return Math.max(0, balls);
  if (balls <= 0) return 0;
  const p = 1 / bins;
  let cdf = 0; // F(j) = P(X ≤ j)
  let pmf = Math.pow(1 - p, balls); // P(X = 0)
  let eMax = 0;
  for (let j = 0; j <= balls; j++) {
    cdf += pmf;
    eMax += 1 - Math.pow(cdf, bins); // term for m = j + 1
    pmf *= ((balls - j) / (j + 1)) * (p / (1 - p)); // P(X = j + 1)
  }
  return eMax;
}

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
  /**
   * Total routed-expert parameters in billions (HF `routed_expert_params_b`).
   * With `numExperts` and `topK`, lets the model charge expert-parallel load
   * imbalance at batch-1; omit to fall back to the balanced active/tp read.
   */
  routedExpertParamsB?: number | null;
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
  /**
   * Serially-dependent compute ops per layer on the critical path. Defaults to
   * a structural estimate (see {@link DEFAULT_DECODE_COMPUTE_OPS_PER_LAYER}),
   * chosen by dense-vs-MoE when omitted.
   */
  computeOpsPerLayer?: number;
  /**
   * Expert-parallel degree. Used (with the model's expert dims) to charge
   * batch-1 expert load imbalance. Defaults to `tp` for MoE, 1 otherwise.
   */
  ep?: number;
  /**
   * Pipeline-parallel degree. Each token crosses `pp − 1` inter-stage send/recv
   * hops over the interconnect. Defaults to 1 (no pipeline parallelism).
   */
  pp?: number;
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
    ppSendRecvS: number;
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

  const ep = params.ep ?? (model.isMoe ? tp : 1);

  // 1. Weight read. The balanced roofline streams the active weights sharded
  // evenly across tp GPUs. The *actual* read additionally charges expert-parallel
  // load imbalance: at batch-1 the token's top-k expert GEMVs land balls-in-bins
  // across the EP GPUs (each expert read in full, EP-sharded not TP-sharded), so
  // the busiest GPU reads E[max] experts, not the balanced top-k/ep. Attention
  // (TP-sharded) keeps the even /tp read.
  const activeBytes = model.activeParamsB * model.bytesPerParam * BYTES_PER_GB;
  const balancedWeightReadS = activeBytes / tp / hbmBytesPerS;

  let weightReadS = balancedWeightReadS;
  const topK = model.topK ?? 0;
  if (
    model.isMoe &&
    ep > 1 &&
    topK > 0 &&
    model.numExperts != null &&
    model.numExperts > 0 &&
    model.routedExpertParamsB != null
  ) {
    const perExpertBytes = (model.routedExpertParamsB / model.numExperts) * model.bytesPerParam * BYTES_PER_GB;
    const expertActiveBytes = Math.min(topK * perExpertBytes, activeBytes);
    const attentionBytes = activeBytes - expertActiveBytes;
    // Busiest EP GPU reads E[max] full experts; attention stays TP-sharded.
    const busiestExperts = expectedMaxBinLoad(topK, ep);
    weightReadS = (attentionBytes / tp + busiestExperts * perExpertBytes) / hbmBytesPerS;
  }

  // 2. TP all-reduce: one all-reduce of the hidden activation per layer. The
  // factor 2 is the ring all-reduce data-volume factor (~2·(N−1)/N ≈ 2).
  let tpAllReduceS = 0;
  if (tp > 1) {
    const allReduceBytes = 2 * model.hiddenSize * activationBytes;
    tpAllReduceS = model.numLayers * (allReduceBytes / interconnectBytesPerS + collectiveLatencyS);
  }

  // 3. EP all-to-all (MoE only): each layer does TWO all-to-alls per token — a
  // dispatch (scatter the hidden vector to its top-k experts) and a combine
  // (gather the expert outputs back). Both are on the serial critical path.
  let epAllToAllS = 0;
  if (model.isMoe && tp > 1 && model.topK !== null && (model.topK ?? 0) > 0) {
    const dispatchBytes = model.topK * model.hiddenSize * activationBytes;
    const perAllToAllS = dispatchBytes / interconnectBytesPerS + collectiveLatencyS;
    epAllToAllS = model.numLayers * 2 * perAllToAllS;
  }

  // 4. Op-chain latency: the dominant batch-1 term. Each layer runs a sequence
  // of serially-dependent compute kernels (collectives counted above), each
  // paying the per-op launch/dispatch latency with nothing to overlap it.
  const opsPerLayer =
    params.computeOpsPerLayer ??
    (model.isMoe
      ? DEFAULT_DECODE_COMPUTE_OPS_PER_LAYER.moe
      : DEFAULT_DECODE_COMPUTE_OPS_PER_LAYER.dense);
  const launchS = launchLatencyS * model.numLayers * opsPerLayer;

  // 5. Pipeline send/recv: a token crosses pp − 1 stage boundaries, each a
  // point-to-point transfer of its hidden activation over the interconnect.
  const pp = Math.max(1, params.pp ?? 1);
  let ppSendRecvS = 0;
  if (pp > 1) {
    const hopBytes = model.hiddenSize * activationBytes;
    ppSendRecvS = (pp - 1) * (hopBytes / interconnectBytesPerS + collectiveLatencyS);
  }

  const perTokenS = weightReadS + tpAllReduceS + epAllToAllS + launchS + ppSendRecvS;

  return {
    perTokenS,
    singleStreamTokS: 1 / perTokenS,
    // Roofline stays the balanced weight-read ceiling (imbalance/comm sit below).
    bandwidthRooflineTokS: 1 / balancedWeightReadS,
    breakdown: { weightReadS, tpAllReduceS, epAllToAllS, launchS, ppSendRecvS },
  };
}
