/**
 * Column model for the Performance persona.
 *
 * The persona no longer fans a model across four concurrency tiers (Single
 * Agent … Agent Swarm). Instead every model is shown at two operating points,
 * both sized at the fixed 90% GPU-memory utilization the calculator assumes:
 *
 *   - "fit"   — the smallest (cheapest) GPU setup that fits the model and can
 *               serve at least one stream. The entry point.
 *   - "scale" — the cheapest GPU setup that sustains SCALE_STREAM_TARGET+
 *               concurrent streams. The "keep the GPU busy" point.
 */

export type PerformanceColumnKey = "fit" | "scale";

/** Concurrency floor a "scale" setup must sustain. */
export const SCALE_STREAM_TARGET = 100;

export interface PerformanceColumnConfig {
  key: PerformanceColumnKey;
  /** Minimum operating streams the chosen setup must admit. */
  minStreams: number;
  label: string;
  description: string;
}

export const PERFORMANCE_COLUMNS: PerformanceColumnConfig[] = [
  {
    key: "fit",
    minStreams: 1,
    label: "Fit",
    description: "Cheapest configuration that fits the model",
  },
  {
    key: "scale",
    minStreams: SCALE_STREAM_TARGET,
    label: "Scale",
    description: `Cheapest configuration that supports ${SCALE_STREAM_TARGET}+ streams`,
  },
];
