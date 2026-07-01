import type { DeploymentEstimate, InterconnectTier, ThroughputState } from "@/types";
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
}): string {
  const fmtTokens = (n: number) =>
    n >= 1000 ? `${Math.round(n / 1000)}K` : `${n}`;
  return `${fmtTokens(context.avgInputTokens)} in / ${fmtTokens(
    context.avgOutputTokens,
  )} out`;
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

const THROUGHPUT_UNMODELED_TOOLTIP: Record<Exclude<ThroughputState, "modeled">, string> = {
  "unsupported-arch":
    "Throughput is not modeled for this architecture (linear-attention / SSM hybrid or sparse attention) — the decode-latency model assumes a uniform full-attention transformer. Operating streams and cost are still accurate (VRAM accounting).",
  "data-incomplete":
    "Throughput needs model dims (hidden_size) that aren't populated yet. Operating streams and cost are still accurate (VRAM accounting).",
};

/** Badge shown in place of the source badge when throughput isn't modeled. */
export function ThroughputUnmodeledBadge({ state }: { state: Exclude<ThroughputState, "modeled"> }) {
  return (
    <Badge
      variant="outline"
      className="text-[10px] font-normal cursor-help text-muted-foreground"
      title={THROUGHPUT_UNMODELED_TOOLTIP[state]}
    >
      {state === "unsupported-arch" ? "not modeled" : "dims pending"}
    </Badge>
  );
}

/**
 * Read-only panel for one offering's {@link DeploymentEstimate}. Operating
 * streams + cost are robust for every architecture and always shown; the
 * throughput numbers (single-stream / aggregate tok/s) are shown only when the
 * architecture is modeled, else a "not modeled" badge takes their place.
 */
export function DeploymentEstimatePanel({
  estimate,
  className,
}: {
  estimate: DeploymentEstimate;
  className?: string;
}) {
  const throughputModeled =
    estimate.throughputState === "modeled" && estimate.singleStreamTokS !== null;

  return (
    <div className={cn("space-y-0.5", className)}>
      {/* Lead line: single-stream tok/s when modeled, else the robust streams band. */}
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
        {throughputModeled ? (
          <>
            <span className="text-sm font-medium text-foreground tabular-nums">
              {formatTokS(estimate.singleStreamTokS as number)}
            </span>
            <span className="text-xs text-muted-foreground">single-stream</span>
          </>
        ) : (
          <>
            <span className="text-sm font-medium text-foreground tabular-nums">
              {formatStreamBand(estimate.operatingStreams)}
            </span>
            <span className="text-xs text-muted-foreground">operating streams</span>
            <ThroughputUnmodeledBadge
              state={estimate.throughputState === "modeled" ? "data-incomplete" : estimate.throughputState}
            />
          </>
        )}
      </div>

      {/* Detail line: streams + aggregate when modeled, else a short note. */}
      {throughputModeled ? (
        <div className="text-xs text-muted-foreground tabular-nums">
          {formatStreamBand(estimate.operatingStreams)} operating streams
          {estimate.aggregateTokS !== null && (
            <>
              {" · "}
              {formatTokS(estimate.aggregateTokS)} aggregate
            </>
          )}
        </div>
      ) : (
        <div className="text-xs text-muted-foreground">
          throughput not modeled for this architecture
        </div>
      )}

      {/* Key assumptions: context size, interconnect tier, MoE-vs-dense. */}
      <div className="text-[10px] leading-tight text-muted-foreground/70">
        {describeAssumptions(estimate).join(" · ")}
      </div>
    </div>
  );
}
