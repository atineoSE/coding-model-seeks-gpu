"use client";

import type { Persona, MatrixCell, Model, GpuSetupOption } from "@/types";
import { PERFORMANCE_COLUMNS } from "@/lib/performance-columns";
import {
  getModelMemory,
  resolveModelPrecision,
  WEIGHT_OVERHEAD_FACTOR,
} from "@/lib/calculations";
import { formatTokS, interconnectBadgeLabel } from "@/components/deployment-estimate-panel";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useIsDesktop } from "@/hooks/use-media-query";
import { formatModelName } from "@/lib/utils";
import { ExternalLink } from "lucide-react";

interface RecommendationMatrixProps {
  rows: MatrixCell[][];
  persona: Persona;
  currencySymbol?: string;
  sotaTotalBenchmarkCost?: number | null;
  benchmarkDisplayName?: string;
}

const RANK_COLORS = [
  "bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-500/30",
  "bg-slate-400/10 text-slate-600 dark:text-slate-400 border-slate-400/30",
  "bg-orange-600/10 text-orange-700 dark:text-orange-400 border-orange-600/30",
];

function formatCurrency(value: number, symbol: string = "$"): string {
  if (value >= 1000) {
    return `${symbol}${(value / 1000).toFixed(1)}k`;
  }
  return `${symbol}${Math.round(value)}`;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Maps percentOfSota (0–1) to a color anchored at the extremes:
 * - 100% → vibrant green (hue 120°, full saturation)
 * - 0%   → vibrant red  (hue 0°,   full saturation)
 * - midpoint → desaturated, so the "washed out" effect is in saturation,
 *   not hue — an 80% model stays recognizably green, just slightly muted.
 */
function sotaColor(percent: number): string {
  const p = Math.max(0, Math.min(1, percent));
  const hue = Math.round(p * 120); // 0=red … 120=green
  // Full saturation at the extremes, drained toward the midpoint
  const saturation = Math.round(30 + 45 * Math.abs(2 * p - 1)); // 30% at 50%, 75% at 0/100%
  return `hsl(${hue}, ${saturation}%, 42%)`;
}

function SotaBarCell({
  percentOfSota,
  prevPercentOfSota,
  nextPercentOfSota,
  isFirst,
  isLast,
}: {
  percentOfSota: number | null;
  prevPercentOfSota: number | null;
  nextPercentOfSota: number | null;
  isFirst: boolean;
  isLast: boolean;
}) {
  // Unranked rows have no score — drop the SOTA bar entirely (the surrounding
  // w-8 column keeps the spacing so rows still align with the ranked table).
  if (percentOfSota === null) return null;

  const color = sotaColor(percentOfSota);
  // Midpoint colors at each row boundary — used as the shared meeting point so
  // the bottom of row N and the top of row N+1 start/end at the exact same color.
  const midFromPrev = prevPercentOfSota !== null
    ? sotaColor((prevPercentOfSota + percentOfSota) / 2)
    : color;
  const midToNext = nextPercentOfSota !== null
    ? sotaColor((percentOfSota + nextPercentOfSota) / 2)
    : color;

  return (
    <div className="absolute inset-0 flex items-center justify-center">
      {/* top half: from shared mid-boundary above up to this dot */}
      {!isFirst && (
        <div
          className="absolute w-[4px] left-1/2 -translate-x-1/2 top-0"
          style={{
            height: "50%",
            background: `linear-gradient(to bottom, ${midFromPrev}, ${color})`,
          }}
        />
      )}
      {/* bottom half: from this dot down to shared mid-boundary below */}
      {!isLast && (
        <div
          className="absolute w-[4px] left-1/2 -translate-x-1/2 bottom-0"
          style={{
            height: "50%",
            background: `linear-gradient(to bottom, ${color}, ${midToNext})`,
          }}
        />
      )}
      {/* dot */}
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className="relative z-10 w-2 h-2 rounded-full cursor-help shrink-0"
            style={{ backgroundColor: color }}
          />
        </TooltipTrigger>
        <TooltipContent>{formatPercent(percentOfSota)} of SOTA</TooltipContent>
      </Tooltip>
    </div>
  );
}

function isMixedPrecision(model: Model): boolean {
  return model.routed_expert_params_b !== null;
}

function formatWeightMemory(model: Model): string {
  const precision = resolveModelPrecision(model);
  const memGb = getModelMemory(model, precision);
  if (memGb === null) return "";
  return `${Math.round(memGb)} GB`;
}

function getMinVramGb(model: Model): number | null {
  const precision = resolveModelPrecision(model);
  const memGb = getModelMemory(model, precision);
  if (memGb === null) return null;
  return Math.ceil(memGb * WEIGHT_OVERHEAD_FACTOR);
}

function PrecisionBadge({ model }: { model: Model }) {
  if (!model.precision) return null;

  if (isMixedPrecision(model)) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant="secondary" className="text-[10px] cursor-help">
            {model.precision}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>
          QAT INT4 experts, BF16 attention — {formatWeightMemory(model)} weights
        </TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Badge variant="secondary" className="text-[10px]">
      {model.precision}
    </Badge>
  );
}

/**
 * Human labels for the attention family. Rendered as a subtle badge — the only
 * cue in the UI about which architectures we model single-stream / aggregate
 * throughput for (GQA and MLA are modeled; the sparse families are not).
 */
const ATTENTION_LABELS: Record<NonNullable<Model["attention_type"]>, { short: string; full: string }> = {
  GQA: { short: "GQA", full: "Grouped-Query Attention" },
  MLA: { short: "MLA", full: "Multi-head Latent Attention" },
  DSV4: { short: "DSA", full: "DeepSeek Sparse Attention" },
  MSA: { short: "MSA", full: "MiniMax Sparse Attention" },
};

function AttentionBadge({ model }: { model: Model }) {
  if (!model.attention_type) return null;
  const label = ATTENTION_LABELS[model.attention_type];
  if (!label) return null;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="secondary" className="text-[10px] cursor-help">
          {label.short}
        </Badge>
      </TooltipTrigger>
      <TooltipContent>{label.full}</TooltipContent>
    </Tooltip>
  );
}

function LicenseBadge({ model }: { model: Model }) {
  if (!model.license_name) return null;
  const badge = (
    <Badge
      variant="outline"
      className={`text-[10px] ${model.license_url ? "hover:bg-accent cursor-pointer" : ""}`}
    >
      {model.license_name}
    </Badge>
  );
  return model.license_url ? (
    <a href={model.license_url} target="_blank" rel="noopener noreferrer">
      {badge}
    </a>
  ) : (
    badge
  );
}

function ModelInfo({ cell, rowIdx }: { cell: MatrixCell; rowIdx: number }) {
  const { model } = cell;
  const minVram = getMinVramGb(model);
  const modelUrl = model.hf_model_id
    ? `https://huggingface.co/${model.hf_model_id}`
    : model.model_url ?? null;

  return (
    <div className="flex items-start gap-2">
      {/* Rank medal only for ranked rows — unranked rows are ordered by size,
          not quality, so a #N medal would be misleading. */}
      {!cell.isUnranked && (
        <Badge
          variant="outline"
          className={`shrink-0 text-xs font-bold ${RANK_COLORS[rowIdx] ?? ""}`}
        >
          #{rowIdx + 1}
        </Badge>
      )}
      <div className="min-w-0 space-y-1">
        {/* Model name — HF link, fallback URL, or plain text */}
        <div className="font-semibold text-sm truncate">
          {modelUrl ? (
            <a
              href={modelUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:underline inline-flex items-center gap-1"
            >
              {formatModelName(model.model_name)}
              <ExternalLink className="size-3 text-muted-foreground" />
            </a>
          ) : (
            formatModelName(model.model_name)
          )}
        </div>

        {/* License — its own dedicated line */}
        {model.license_name && (
          <div>
            <LicenseBadge model={model} />
          </div>
        )}

        {/* Params (+ active for MoE) + min VRAM + context */}
        {(model.learnable_params_b !== null ||
          minVram !== null ||
          model.context_length !== null) && (
          <div className="text-xs text-muted-foreground">
            {model.learnable_params_b !== null && (
              <>
                {Math.round(model.learnable_params_b)}b
                {model.architecture === "MoE" && model.active_params_b !== null && (
                  <> ({Math.round(model.active_params_b)}b active)</>
                )}
              </>
            )}
            {minVram !== null && (
              <>{model.learnable_params_b !== null ? " · " : ""}min {minVram} GB VRAM</>
            )}
            {model.context_length !== null && (
              <>
                {model.learnable_params_b !== null || minVram !== null ? " · " : ""}
                {Math.round(model.context_length / 1024)}K context
              </>
            )}
          </div>
        )}

        {/* SOTA percentage + API cost — or an explicit gap for unranked models */}
        <div className="text-xs text-muted-foreground">
          {cell.isUnranked || cell.percentOfSota === null ? (
            "Unranked"
          ) : (
            <>
              {formatPercent(cell.percentOfSota)} of SOTA
              {cell.totalBenchmarkCost !== null && (
                <> · {formatCurrency(cell.totalBenchmarkCost)} in API costs</>
              )}
            </>
          )}
        </div>

        {/* Architecture badges — MoE, precision, attention family */}
        <div className="flex flex-wrap items-center gap-1">
          {model.architecture === "MoE" && (
            <Badge variant="secondary" className="text-[10px]">
              MoE
            </Badge>
          )}
          <PrecisionBadge model={model} />
          <AttentionBadge model={model} />
        </div>
      </div>
    </div>
  );
}

/**
 * One Performance cell: the chosen GPU setup rendered as a compact card —
 * GPU layout, monthly price, operating streams, and (when the architecture is
 * modeled) single-stream and aggregate throughput. No cost-per-stream,
 * utilization, or "modeled/unmodeled" badges — missing throughput is simply
 * omitted.
 */
function CellContent({ cell, currencySymbol = "$" }: { cell: MatrixCell; currencySymbol?: string }) {
  if (cell.exceedsCapacity) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground/50 text-xs italic">
        Exceeds capacity
      </div>
    );
  }

  const setup: GpuSetupOption | null = cell.gpuSetups[0] ?? null;
  if (!setup) return null;
  const est = setup.deploymentEstimate;
  const interconnectBadge = interconnectBadgeLabel(setup.gpuName, setup.gpuCount);

  return (
    <div className="space-y-1">
      {/* GPU layout */}
      <div className="text-sm font-medium">
        {setup.gpuCount}× {setup.gpuName}
        {setup.isProjected && " (*)"}
        {interconnectBadge && (
          <span className="text-muted-foreground font-normal ml-1.5">{interconnectBadge}</span>
        )}
      </div>

      {/* Monthly price */}
      <div className="text-sm text-muted-foreground">
        {formatCurrency(setup.monthlyCost, currencySymbol)}/mo
      </div>

      {/* Operating streams — whole-prompt admission (KV for the full context per stream) */}
      <div className="text-xs text-muted-foreground">{setup.maxConcurrentStreams} streams</div>

      {/* Single-stream throughput — omitted when not modeled */}
      {est?.singleStreamTokS != null && (
        <div className="text-xs text-muted-foreground">
          {formatTokS(est.singleStreamTokS)} / stream
        </div>
      )}

      {/* Aggregate throughput — omitted when not modeled */}
      {est?.aggregateTokS != null && (
        <div className="text-xs text-muted-foreground">
          {formatTokS(est.aggregateTokS)} aggregate
        </div>
      )}
    </div>
  );
}

/**
 * Heatmap steps using Tailwind's built-in palette (green → red).
 * Each entry is a class string with light + dark variants.
 */
const HEATMAP_STEPS = [
  "bg-green-100 dark:bg-green-900/30",
  "bg-lime-100 dark:bg-lime-900/30",
  "bg-yellow-100 dark:bg-yellow-900/30",
  "bg-orange-100 dark:bg-orange-900/30",
  "bg-red-100 dark:bg-red-900/30",
];

/**
 * Pick a heatmap class for a cell based on its monthly cost relative to the
 * column's min/max: green = cheapest, red = most expensive.
 */
function getCellHeatmapClass(
  value: number | null,
  minVal: number,
  maxVal: number,
  exceedsCapacity: boolean,
): string {
  if (value === null || exceedsCapacity || minVal >= maxVal) return "";

  let t = (value - minVal) / (maxVal - minVal);
  t = Math.min(1, Math.max(0, t));
  const idx = Math.min(HEATMAP_STEPS.length - 1, Math.floor(t * HEATMAP_STEPS.length));

  return HEATMAP_STEPS[idx];
}

/** The monthly cost that drives a cell's heatmap color (null when no setup). */
function cellMonthlyCost(cell: MatrixCell): number | null {
  return cell.gpuSetups[0]?.monthlyCost ?? null;
}

function SotaHeader({
  cell,
  currencySymbol,
  sotaTotalBenchmarkCost,
  benchmarkDisplayName,
}: {
  cell: MatrixCell | undefined;
  currencySymbol: string;
  sotaTotalBenchmarkCost?: number | null;
  benchmarkDisplayName?: string;
}) {
  const sotaScore = cell?.sotaScore ?? null;
  if (!sotaScore) return null;
  return (
    <div className="text-xs font-normal">
      <span className="font-semibold text-foreground">
        SOTA: {formatModelName(sotaScore.sota_model_name)}
      </span>{" "}
      <span className="font-mono">({sotaScore.sota_score.toFixed(1)})</span>
      {sotaTotalBenchmarkCost !== null && sotaTotalBenchmarkCost !== undefined && (
        <>
          {" "}&mdash;{" "}
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="cursor-help border-b border-dotted border-muted-foreground/50">
                {formatCurrency(sotaTotalBenchmarkCost, currencySymbol)} in API costs
                {benchmarkDisplayName && <> for {benchmarkDisplayName}</>}
              </span>
            </TooltipTrigger>
            <TooltipContent>
              Cost of evaluating this model on the selected benchmark via API,
              as reported by the OpenHands Index.
            </TooltipContent>
          </Tooltip>
        </>
      )}
    </div>
  );
}

function UtilizationNote() {
  return (
    <p className="text-xs text-muted-foreground mt-3">
      Streams and throughput are sized at 90% GPU memory utilization, using the
      context window from Advanced settings.
    </p>
  );
}

function MobileMatrixView({
  rows,
  currencySymbol = "$",
  colMin,
  colMax,
  sotaTotalBenchmarkCost,
  benchmarkDisplayName,
}: {
  rows: MatrixCell[][];
  currencySymbol: string;
  colMin: number[];
  colMax: number[];
  sotaTotalBenchmarkCost?: number | null;
  benchmarkDisplayName?: string;
}) {
  return (
    <div className="space-y-3">
      <div className="mb-1">
        <SotaHeader
          cell={rows[0]?.[0]}
          currencySymbol={currencySymbol}
          sotaTotalBenchmarkCost={sotaTotalBenchmarkCost}
          benchmarkDisplayName={benchmarkDisplayName}
        />
      </div>
      {rows.map((row, rowIdx) => {
        const cell0 = row[0];
        if (!cell0) return null;
        return (
          <div key={rowIdx} className="rounded-lg border overflow-hidden">
            <div className="flex">
              {/* SOTA gradient left strip — dropped for unranked (no score),
                  but the 4px column is kept so cards still align. */}
              <div
                className="w-[4px] shrink-0"
                style={{
                  backgroundColor:
                    cell0.percentOfSota === null
                      ? "transparent"
                      : sotaColor(cell0.percentOfSota),
                }}
              />
              <div className="flex-1 p-3">
                <ModelInfo cell={cell0} rowIdx={rowIdx} />
                <div className="grid grid-cols-2 gap-2 mt-3">
                  {PERFORMANCE_COLUMNS.map((col, colIdx) => {
                    const cell = row[colIdx];
                    if (!cell) return null;
                    const heatmap = getCellHeatmapClass(
                      cellMonthlyCost(cell), colMin[colIdx], colMax[colIdx], cell.exceedsCapacity,
                    );
                    return (
                      <div
                        key={col.key}
                        className={`rounded-md border p-2 ${cell.exceedsCapacity ? "bg-muted/20" : heatmap}`}
                      >
                        <div className="text-xs font-medium text-muted-foreground mb-1">
                          {col.label}
                        </div>
                        <div className="tabular-nums">
                          <CellContent cell={cell} currencySymbol={currencySymbol} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function RecommendationMatrix({
  rows,
  currencySymbol = "$",
  sotaTotalBenchmarkCost,
  benchmarkDisplayName,
}: RecommendationMatrixProps) {
  const isDesktop = useIsDesktop();

  if (rows.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No models available for this configuration.
      </div>
    );
  }

  // Per-column heatmap ranges over monthly cost (green = cheapest).
  const numCols = rows[0]?.length ?? 0;
  const colMin: number[] = new Array(numCols).fill(Infinity);
  const colMax: number[] = new Array(numCols).fill(-Infinity);
  for (const row of rows) {
    for (let c = 0; c < row.length; c++) {
      const cell = row[c];
      const val = cellMonthlyCost(cell);
      if (val !== null && !cell.exceedsCapacity) {
        colMin[c] = Math.min(colMin[c], val);
        colMax[c] = Math.max(colMax[c], val);
      }
    }
  }

  // Check if any GPU setup across the matrix is projected
  const hasProjected = rows.some((row) =>
    row.some((cell) => cell.gpuSetups.some((s) => s.isProjected)),
  );

  if (!isDesktop) {
    return (
      <TooltipProvider>
        <MobileMatrixView
          rows={rows}
          currencySymbol={currencySymbol}
          colMin={colMin}
          colMax={colMax}
          sotaTotalBenchmarkCost={sotaTotalBenchmarkCost}
          benchmarkDisplayName={benchmarkDisplayName}
        />
        <UtilizationNote />
        {hasProjected && (
          <p className="text-xs text-muted-foreground mt-1">
            (*) Projected setup, not actually found in GPUs available.
          </p>
        )}
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <div className="overflow-x-auto relative">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="w-8 p-0" />
              <th className="text-left p-3 text-sm font-medium text-muted-foreground w-[240px]">
                <div>Model</div>
                <SotaHeader
                  cell={rows[0]?.[0]}
                  currencySymbol={currencySymbol}
                  sotaTotalBenchmarkCost={sotaTotalBenchmarkCost}
                  benchmarkDisplayName={benchmarkDisplayName}
                />
              </th>
              {PERFORMANCE_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  className="text-center p-3 text-sm font-medium text-muted-foreground align-top"
                >
                  <div className="text-foreground">{col.label}</div>
                  <div className="text-xs font-normal mt-0.5 max-w-[200px] mx-auto">
                    {col.description}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => {
              const cell0 = row[0];
              if (!cell0) return null;

              return (
                <tr
                  key={rowIdx}
                  className="border-t border-border hover:bg-muted/30 transition-colors"
                >
                  {/* SOTA gradient bar column */}
                  <td className="w-8 p-0 relative">
                    <SotaBarCell
                      percentOfSota={cell0.percentOfSota}
                      prevPercentOfSota={rows[rowIdx - 1]?.[0]?.percentOfSota ?? null}
                      nextPercentOfSota={rows[rowIdx + 1]?.[0]?.percentOfSota ?? null}
                      isFirst={rowIdx === 0}
                      isLast={rowIdx === rows.length - 1}
                    />
                  </td>

                  {/* Model info column */}
                  <td className="p-3 align-top">
                    <ModelInfo cell={cell0} rowIdx={rowIdx} />
                  </td>

                  {/* Fit / Scale columns */}
                  {row.map((cell, colIdx) => {
                    const heatmap = getCellHeatmapClass(
                      cellMonthlyCost(cell), colMin[colIdx], colMax[colIdx], cell.exceedsCapacity,
                    );

                    return (
                      <td
                        key={colIdx}
                        className={`p-3 text-center align-middle ${
                          cell.exceedsCapacity ? "bg-muted/20" : heatmap
                        }`}
                      >
                        <CellContent cell={cell} currencySymbol={currencySymbol} />
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
        <UtilizationNote />
        {hasProjected && (
          <p className="text-xs text-muted-foreground mt-1">
            (*) Projected setup, not actually found in GPUs available.
          </p>
        )}
      </div>
    </TooltipProvider>
  );
}
