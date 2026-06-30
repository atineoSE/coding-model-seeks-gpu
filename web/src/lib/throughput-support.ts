/**
 * Whether the calculator models single-stream / aggregate THROUGHPUT for a model.
 *
 * Operating-streams and cost-per-stream are pure VRAM accounting and robust for
 * every architecture (the only architecture input is KV-bytes-per-token). The
 * decode-latency model, by contrast, assumes a uniform full-attention transformer
 * stack — so throughput is only trustworthy for standard MoE/Dense models with
 * GQA or MLA attention. This predicate gates the throughput overlay; it never
 * gates streams/cost, which stay available for all models.
 *
 * Three states, with the reason preserved so the UI can distinguish "we don't
 * model this architecture" (won't change) from "dims not populated yet" (fixable):
 *
 *   - "modeled"          throughput is computed and shown.
 *   - "unsupported-arch" architecture the decode model can't represent:
 *                          · linear-attention / SSM hybrids (only some layers
 *                            attend: num_kv_layers < num_hidden_layers)
 *                          · sparse attention (DSV4, MSA) — indexer/selection
 *                            compute and sub-quadratic context are unmodeled.
 *   - "data-incomplete"  supported architecture, but missing the dims the
 *                          latency model needs (hidden_size).
 */

import type { Model } from "@/types";

export type ThroughputState = "modeled" | "unsupported-arch" | "data-incomplete";

/** Sparse-attention families whose decode/long-context mechanics are unmodeled. */
const SPARSE_ATTENTION_TYPES: ReadonlySet<Model["attention_type"]> = new Set(["DSV4", "MSA"]);

/**
 * True when only a subset of layers carry KV — the signature of a linear-attention
 * / SSM hybrid (Mamba / Gated-DeltaNet mixers interleaved with full attention).
 * The decode model would treat every layer as full attention, so throughput is wrong.
 */
export function isLinearAttentionHybrid(model: Model): boolean {
  return (
    model.num_kv_layers != null &&
    model.num_hidden_layers != null &&
    model.num_kv_layers < model.num_hidden_layers
  );
}

/**
 * Classify a model's throughput modeling. Architectural reasons take precedence
 * over data gaps (an unsupported architecture stays unsupported even if dims are
 * missing).
 */
export function throughputState(model: Model): ThroughputState {
  // Architecture we cannot model — won't change by populating dims.
  if (isLinearAttentionHybrid(model)) return "unsupported-arch";
  if (SPARSE_ATTENTION_TYPES.has(model.attention_type)) return "unsupported-arch";

  // Supported architecture, but the latency model needs these public dims.
  if (model.hidden_size == null || model.num_hidden_layers == null) return "data-incomplete";

  return "modeled";
}

/** Convenience: is throughput shown for this model? */
export function isThroughputModeled(model: Model): boolean {
  return throughputState(model) === "modeled";
}
