import type { DeploymentEstimate, InterconnectTier } from "@/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ============================================================================
// Pure formatting helpers (no React) — exported for unit testing.
//
// These render a read-only DeploymentEstimate (produced by the matrix
// calculator). They never recompute any throughput number; they only format the
// figures the estimate already carries.
// ============================================================================

/** Format a tok/s figure, switching to a "k" suffix for large aggregates. */
export function formatTokS(value: number): string {
  if (value >= 1000) {
    const k = value / 1000;
    return `${k >= 100 ? Math.round(k) : k.toFixed(1)}k tok/s`;
  }
  return `${Math.round(value)} tok/s`;
}

/** Render the operating-streams concurrency as a low–high band (en dash). */
export function formatStreamBand(band: { low: number; high: number }): string {
  if (band.low === band.high) return `${band.low}`;
  return `${band.low}–${band.high}`;
}

/** Human label for the inter-GPU interconnect tier assumption. */
export function formatInterconnectTier(tier: InterconnectTier): string {
  switch (tier) {
    case "nvswitch":
      return "NVSwitch";
    case "nvlink_paired":
      return "NVLink";
    case "none":
      return "PCIe";
  }
}

/** Compact label for the context-window assumption. */
export function formatContextAssumption(context: {
  avgInputTokens: number;
  avgOutputTokens: number;
  prefixReuse: number;
}): string {
  const fmtTokens = (n: number) =>
    n >= 1000 ? `${Math.round(n / 1000)}K` : `${n}`;
  return `${fmtTokens(context.avgInputTokens)} in / ${fmtTokens(
    context.avgOutputTokens,
  )} out · ${Math.round(context.prefixReuse * 100)}% prefix reuse`;
}

/** The full set of key-assumption labels surfaced alongside the estimate. */
export function describeAssumptions(estimate: DeploymentEstimate): string[] {
  return [
    formatContextAssumption(estimate.assumptions.context),
    `${formatInterconnectTier(estimate.assumptions.interconnectTier)} interconnect`,
    estimate.assumptions.moe ? "MoE (active params)" : "Dense",
  ];
}

// ============================================================================
// Read-only display
// ============================================================================

/**
 * Where an estimate's numbers come from. Today every estimate is "modeled"
 * (first-principles). The "measured" slot is reserved for a future stage that
 * overlays real benchmark numbers — the badge already renders it so callers can
 * switch source without UI churn.
 */
export type EstimateSource = "modeled" | "measured";

const ESTIMATE_SOURCE_LABEL: Record<EstimateSource, string> = {
  modeled: "modeled",
  measured: "measured",
};

const ESTIMATE_SOURCE_TOOLTIP: Record<EstimateSource, string> = {
  modeled:
    "First-principles estimate from GPU datasheet specs, model architecture and the parallelism layout — not a benchmark.",
  measured: "Measured from a real benchmark run.",
};

export function EstimateSourceBadge({
  source = "modeled",
}: {
  source?: EstimateSource;
}) {
  return (
    <Badge
      variant="outline"
      className="text-[10px] font-normal cursor-help"
      title={ESTIMATE_SOURCE_TOOLTIP[source]}
    >
      {ESTIMATE_SOURCE_LABEL[source]}
    </Badge>
  );
}

/**
 * Read-only panel for one offering's {@link DeploymentEstimate}: the three
 * developer-facing numbers (single-stream tok/s, operating-streams band,
 * aggregate tok/s) plus the key assumptions and a source badge.
 */
export function DeploymentEstimatePanel({
  estimate,
  source = "modeled",
  className,
}: {
  estimate: DeploymentEstimate;
  source?: EstimateSource;
  className?: string;
}) {
  return (
    <div className={cn("space-y-0.5", className)}>
      {/* Single-stream tok/s — the one-developer interactive feel. */}
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
        <span className="text-sm font-medium text-foreground tabular-nums">
          {formatTokS(estimate.singleStreamTokS)}
        </span>
        <span className="text-xs text-muted-foreground">single-stream</span>
        <EstimateSourceBadge source={source} />
      </div>

      {/* Operating-streams band + aggregate (bonus) throughput. */}
      <div className="text-xs text-muted-foreground tabular-nums">
        {formatStreamBand(estimate.operatingStreams)} operating streams
        {" · "}
        {formatTokS(estimate.aggregateTokS)} aggregate
      </div>

      {/* Key assumptions: context size, interconnect tier, MoE-vs-dense. */}
      <div className="text-[10px] leading-tight text-muted-foreground/70">
        {describeAssumptions(estimate).join(" · ")}
      </div>
    </div>
  );
}
