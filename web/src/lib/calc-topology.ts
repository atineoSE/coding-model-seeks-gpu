/**
 * Parallelism-layout topology mapping.
 *
 * Pure function mapping an inter-GPU interconnect tier + GPU count + model
 * weight footprint to a *feasible* tensor-/pipeline-parallel layout.
 *
 * The rule set is derived from public topology knowledge, not measurement:
 *
 * - Tensor parallelism (TP) shards every layer's matmuls across GPUs and needs
 *   an all-reduce per layer — a high-bandwidth, latency-sensitive collective.
 *   So TP should only span GPUs that share a high-bandwidth link.
 *     · NVSwitch gives every GPU full all-to-all NVLink bandwidth within the
 *       node (NVIDIA DGX/HGX 8-GPU baseboards), so TP can span the whole node.
 *     · An NVLink bridge connects GPUs only in 2-way pairs (e.g. the A100/RTX
 *       PCIe NVLink bridge), so TP across such a system must stay within a pair.
 *     · PCIe-only systems ("none") can still run TP, but every all-reduce
 *       traverses PCIe — far lower bandwidth / higher latency than NVLink — so
 *       the layout is flagged for the latency model to apply its penalty.
 *
 * - Pipeline parallelism (PP) splits the layer stack into stages and only
 *   passes activations point-to-point between adjacent stages — a small,
 *   latency-tolerant transfer (standard Megatron-LM parallelism guidance).
 *   This is what we use to scale *across* the low-bandwidth boundaries that TP
 *   must not cross: on NVLink-paired systems we force PP across the pairs.
 *
 * No fitted or benchmarked constants live here — only the public structural
 * facts above (pair size = 2, NVSwitch fabric is all-to-all).
 */

import type { InterconnectTier } from "@/types";

/**
 * Size of an NVLink-bridge group. NVLink bridges connect GPUs in 2-way pairs,
 * so tensor parallelism on a paired system is capped at 2.
 */
export const NVLINK_PAIR_SIZE = 2;

export interface TopologyInput {
  /** Inter-GPU interconnect tier of the GPU layout. */
  interconnectTier: InterconnectTier;
  /** Total number of GPUs available in the deployment. */
  gpuCount: number;
  /** Model weight footprint in GB, as deployed (i.e. already quantized). */
  modelSizeGb: number;
  /** Usable VRAM per GPU in GB available for weights + runtime. */
  vramPerGpuGb: number;
}

export interface ParallelismLayout {
  /** Tensor-parallel degree (GPUs cooperating on each layer shard). */
  tp: number;
  /** Pipeline-parallel degree (number of sequential layer stages). */
  pp: number;
  /** GPUs actually used by the layout (`tp * pp`); may be < `gpuCount`. */
  gpusUsed: number;
  /** Interconnect tier the layout was derived for. */
  interconnectTier: InterconnectTier;
  /**
   * True when the tensor-parallel all-reduce crosses a low-bandwidth (PCIe)
   * boundary. The latency model must apply its interconnect penalty when set.
   */
  tpCrossesSlowLink: boolean;
  /** Whether the model weights fit across the `gpusUsed` GPUs. */
  feasible: boolean;
  /** Human-readable explanation of the chosen layout. */
  reason: string;
}

/**
 * Map a topology + model size to a feasible parallelism layout.
 *
 * The layout always uses the most aggressive TP the interconnect tier allows
 * (TP keeps single-stream latency low) and only introduces PP where TP would
 * otherwise have to cross a link it cannot safely span.
 */
export function calcTopology(input: TopologyInput): ParallelismLayout {
  const { interconnectTier, gpuCount, modelSizeGb, vramPerGpuGb } = input;

  if (!Number.isInteger(gpuCount) || gpuCount < 1) {
    throw new Error(`gpuCount must be a positive integer, got ${gpuCount}`);
  }
  if (!(vramPerGpuGb > 0)) {
    throw new Error(`vramPerGpuGb must be positive, got ${vramPerGpuGb}`);
  }
  if (!(modelSizeGb >= 0)) {
    throw new Error(`modelSizeGb must be non-negative, got ${modelSizeGb}`);
  }

  let tp: number;
  let pp: number;
  let tpCrossesSlowLink: boolean;
  let reason: string;

  switch (interconnectTier) {
    case "nvswitch": {
      // All-to-all NVLink fabric: TP can span the whole node, no forced PP.
      tp = gpuCount;
      pp = 1;
      tpCrossesSlowLink = false;
      reason =
        `NVSwitch fabric: tensor-parallel across all ${gpuCount} GPU(s) ` +
        `(no forced pipeline parallelism).`;
      break;
    }
    case "nvlink_paired": {
      // NVLink bridges are 2-way: cap TP at the pair size and pipeline across
      // the pairs. A lone GPU degrades to TP=1.
      tp = Math.min(NVLINK_PAIR_SIZE, gpuCount);
      pp = Math.floor(gpuCount / tp);
      tpCrossesSlowLink = false;
      reason =
        `NVLink-paired: tensor-parallel capped at the pair size (TP=${tp}), ` +
        `pipeline-parallel forced across ${pp} pair(s).`;
      break;
    }
    case "none": {
      // PCIe-only: TP is allowed but every all-reduce crosses PCIe, so flag it
      // for the latency model's interconnect penalty. No forced PP.
      tp = gpuCount;
      pp = 1;
      tpCrossesSlowLink = gpuCount > 1;
      reason =
        `PCIe-only (no NVLink): tensor-parallel across ${gpuCount} GPU(s), ` +
        `flagged for the latency penalty as TP all-reduce traverses PCIe.`;
      break;
    }
    default: {
      // Exhaustiveness guard — keeps the switch honest if the tier union grows.
      const exhaustive: never = interconnectTier;
      throw new Error(`Unknown interconnect tier: ${String(exhaustive)}`);
    }
  }

  const gpusUsed = tp * pp;
  // Weights are sharded evenly across every GPU in the layout, so the model
  // fits iff aggregate VRAM over the used GPUs covers the weight footprint.
  const feasible = gpusUsed * vramPerGpuGb >= modelSizeGb;

  return {
    tp,
    pp,
    gpusUsed,
    interconnectTier,
    tpCrossesSlowLink,
    feasible,
    reason,
  };
}
