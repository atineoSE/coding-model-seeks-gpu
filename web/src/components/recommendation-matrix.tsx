"use client";

import type { Persona, MatrixCell, Model, GpuSetupOption } from "@/types";
import { CONCURRENCY_TIERS } from "@/lib/concurrency-tiers";
import {
  getModelMemory,
  resolveModelPrecision,
  isNvLink,
  WEIGHT_OVERHEAD_FACTOR,
} from "@/lib/calculations";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useIsDesktop } from "@/hooks/use-media-query";
import { ExternalLink } from "lucide-react";

interface RecommendationMatrixProps {
  rows: MatrixCell[][];
  persona: Persona;
  currencySymbol?: string;
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

function isMixedPrecision(model: Model): boolean {
  return model.routed_expert_params_b !== null;
}

function formatWeightMemory(model: Model): string {
  const precision = resolveModelPrecision(model);
  const memGb = getModelMemory(model, precision);
  if (memGb === null) return "";
  return `${Math.round(memGb)} GB`;
}

function formatSotaModelName(name: string): string {
  return name
    .replace(/(\d)-(\d)/g, "$1.$2") // digit-digit hyphens → dots (version numbers)
    .replace(/-/g, " ")
    .replace(/\b[a-z]/g, (c) => c.toUpperCase());
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

function ModelInfo({ cell, rowIdx }: { cell: MatrixCell; rowIdx: number }) {
  const { model } = cell;
  const minVram = getMinVramGb(model);

  return (
    <div className="flex items-start gap-2">
      <Badge
        variant="outline"
        className={`shrink-0 text-xs font-bold ${RANK_COLORS[rowIdx] ?? ""}`}
      >
        #{rowIdx + 1}
      </Badge>
      <div className="min-w-0">
        {/* Model name — HF link or plain text */}
        <div className="font-semibold text-sm truncate">
          {model.hf_model_id ? (
            <a
              href={`https://huggingface.co/${model.hf_model_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:underline inline-flex items-center gap-1"
            >
              {model.model_name}
              <ExternalLink className="size-3 text-muted-foreground" />
            </a>
          ) : (
            model.model_name
          )}
        </div>

        {/* Params + min VRAM */}
        <div className="text-xs text-muted-foreground mt-0.5">
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
        </div>

        {/* SOTA percentage */}
        <div className="text-xs text-muted-foreground mt-0.5">
          {formatPercent(cell.percentOfSota)} of SOTA
        </div>

        {/* Badges */}
        <div className="flex items-center gap-1 mt-1">
          {model.architecture === "MoE" && (
            <Badge variant="secondary" className="text-[10px]">
              MoE
            </Badge>
          )}
          <PrecisionBadge model={model} />
        </div>
      </div>
    </div>
  );
}

function GpuSetupBlock({ setup, currencySymbol = "$" }: { setup: GpuSetupOption; currencySymbol?: string }) {
  return (
    <div>
      <div className="text-sm font-medium">
        {setup.gpuCount}× {setup.gpuName}
        {setup.gpuCount > 1 && isNvLink(setup.interconnect) && (
          <span className="text-muted-foreground font-normal ml-1.5">NVLink</span>
        )}
      </div>
      <div className="text-sm text-muted-foreground">
        {formatCurrency(setup.monthlyCost, currencySymbol)}/mo
      </div>
    </div>
  );
}

function CellContent({ cell, persona, currencySymbol = "$" }: { cell: MatrixCell; persona: Persona; currencySymbol?: string }) {
  if (cell.exceedsCapacity) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground/50 text-xs italic">
        Exceeds capacity
      </div>
    );
  }

  if (persona === "performance") {
    const primary = cell.gpuSetups[0] ?? null;
    const alternatives = cell.gpuSetups.slice(1, 3);
    const hasAlternatives = alternatives.length > 0;

    const cellBody = (
      <div className="space-y-2">
        {/* Primary (cheapest) GPU setup */}
        {primary && <GpuSetupBlock setup={primary} currencySymbol={currencySymbol} />}

        {/* Per-stream metrics: cost + throughput on one line */}
        {(cell.costPerStreamPerMonth !== null || cell.decodeThroughputTokS !== null) && (
          <div className="text-xs text-muted-foreground">
            {cell.costPerStreamPerMonth !== null && (
              <>{formatCurrency(cell.costPerStreamPerMonth, currencySymbol)}/stream</>
            )}
            {cell.costPerStreamPerMonth !== null && cell.decodeThroughputTokS !== null && (
              <> · </>
            )}
            {cell.decodeThroughputTokS !== null && (
              <>{Math.round(cell.decodeThroughputTokS)} tok/s</>
            )}
          </div>
        )}

        {/* Utilization (most subdued) */}
        {cell.utilization !== null && (
          <div className="text-xs text-muted-foreground/50">
            ~{Math.round(cell.utilization * 100)}% utilized
          </div>
        )}
      </div>
    );

    if (!hasAlternatives) return cellBody;

    return (
      <HoverCard openDelay={200} closeDelay={100}>
        <HoverCardTrigger asChild>
          <div className="cursor-default">{cellBody}</div>
        </HoverCardTrigger>
        <HoverCardContent align="center" className="w-auto min-w-48">
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground mb-2">
              Alternative GPU setups
            </div>
            {cell.gpuSetups.slice(0, 3).map((setup, i) => (
              <div
                key={i}
                className={`${i === 0 ? "font-semibold" : "text-muted-foreground"}`}
              >
                <div className="text-sm">
                  {setup.gpuCount}× {setup.gpuName}
                  {setup.gpuCount > 1 && isNvLink(setup.interconnect) && (
                    <span className="font-normal ml-1.5 text-muted-foreground">NVLink</span>
                  )}
                </div>
                <div className="text-xs">
                  {formatCurrency(setup.monthlyCost, currencySymbol)}/mo
                  {" · "}
                  {formatCurrency(setup.costPerStreamPerMonth, currencySymbol)}/stream
                  {setup.decodeThroughputTokS !== null && (
                    <> · {Math.round(setup.decodeThroughputTokS)} tok/s</>
                  )}
                </div>
              </div>
            ))}
          </div>
        </HoverCardContent>
      </HoverCard>
    );
  }

  // Budget persona — hardware already secured, show throughput + utilization only
  return (
    <div className="space-y-1">
      {/* Primary: throughput */}
      {cell.decodeThroughputTokS !== null && (
        <div className="text-sm font-medium">
          {Math.round(cell.decodeThroughputTokS)} tok/s/stream
        </div>
      )}

      {/* Utilization (most subdued) */}
      {cell.utilization !== null && (
        <div className="text-xs text-muted-foreground/50">
          ~{Math.round(cell.utilization * 100)}% utilized
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
 * Pick a heatmap class for a cell based on its value relative to the column's
 * min/max. When higherIsBetter is false (cost): green = lowest, red = highest.
 * When higherIsBetter is true (throughput): green = highest, red = lowest.
 */
function getCellHeatmapClass(
  value: number | null,
  minVal: number,
  maxVal: number,
  exceedsCapacity: boolean,
  higherIsBetter: boolean = false,
): string {
  if (value === null || exceedsCapacity || minVal >= maxVal) return "";

  let t = (value - minVal) / (maxVal - minVal);
  if (higherIsBetter) t = 1 - t;
  t = Math.min(1, Math.max(0, t));
  const idx = Math.min(HEATMAP_STEPS.length - 1, Math.floor(t * HEATMAP_STEPS.length));

  return HEATMAP_STEPS[idx];
}

function MobileMatrixView({ rows, persona, currencySymbol = "$", colMin, colMax, useThroughput }: {
  rows: MatrixCell[][];
  persona: Persona;
  currencySymbol: string;
  colMin: number[];
  colMax: number[];
  useThroughput: boolean;
}) {
  const sotaScore = rows[0]?.[0]?.sotaScore ?? null;

  return (
    <Tabs defaultValue="multi_agent">
      {sotaScore && (
        <div className="text-xs text-muted-foreground mb-2">
          <span className="font-semibold text-foreground">
            SOTA: {formatSotaModelName(sotaScore.sota_model_name)}
          </span>{" "}
          <span className="font-mono">({sotaScore.sota_score.toFixed(1)})</span>
        </div>
      )}
      <TabsList className="grid w-full grid-cols-4 group-data-[orientation=horizontal]/tabs:h-auto">
        {CONCURRENCY_TIERS.map((tier) => (
          <TabsTrigger key={tier.key} value={tier.key} className="text-xs px-1 py-1.5 h-auto whitespace-normal leading-tight">
            {tier.label}
          </TabsTrigger>
        ))}
      </TabsList>

      {CONCURRENCY_TIERS.map((tier, colIdx) => (
        <TabsContent key={tier.key} value={tier.key} className="space-y-3 mt-3">
          <div className="text-xs text-muted-foreground text-center">
            {tier.midpoint} streams &mdash; {tier.description}
          </div>
          {rows.map((row, rowIdx) => {
            const cell0 = row[0];
            const cell = row[colIdx];
            if (!cell0 || !cell) return null;

            const heatmapVal = useThroughput ? cell.decodeThroughputTokS : cell.costPerStreamPerMonth;
            const heatmap = getCellHeatmapClass(
              heatmapVal, colMin[colIdx], colMax[colIdx], cell.exceedsCapacity, useThroughput,
            );

            return (
              <div
                key={rowIdx}
                className={`rounded-lg border p-3 ${cell.exceedsCapacity ? "bg-muted/20" : heatmap}`}
              >
                <ModelInfo cell={cell0} rowIdx={rowIdx} />
                <hr className="my-2 border-border" />
                <div className="tabular-nums">
                  <CellContent cell={cell} persona={persona} currencySymbol={currencySymbol} />
                </div>
              </div>
            );
          })}
        </TabsContent>
      ))}
    </Tabs>
  );
}

export function RecommendationMatrix({ rows, persona, currencySymbol = "$" }: RecommendationMatrixProps) {
  const isDesktop = useIsDesktop();

  if (rows.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No models available for this configuration.
      </div>
    );
  }

  // Extract SOTA info from first row
  const sotaScore = rows[0]?.[0]?.sotaScore ?? null;

  // Compute per-column heatmap ranges.
  // Performance persona: cost-based (green = cheapest).
  // Budget persona: throughput-based (green = highest tok/s).
  const useThroughput = persona === "budget";
  const numCols = rows[0]?.length ?? 0;
  const colMin: number[] = new Array(numCols).fill(Infinity);
  const colMax: number[] = new Array(numCols).fill(-Infinity);
  for (const row of rows) {
    for (let c = 0; c < row.length; c++) {
      const cell = row[c];
      const val = useThroughput ? cell.decodeThroughputTokS : cell.costPerStreamPerMonth;
      if (val !== null && !cell.exceedsCapacity) {
        colMin[c] = Math.min(colMin[c], val);
        colMax[c] = Math.max(colMax[c], val);
      }
    }
  }

  if (!isDesktop) {
    return (
      <TooltipProvider>
        <MobileMatrixView
          rows={rows}
          persona={persona}
          currencySymbol={currencySymbol}
          colMin={colMin}
          colMax={colMax}
          useThroughput={useThroughput}
        />
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <div className="overflow-x-auto relative">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left p-3 text-sm font-medium text-muted-foreground w-[200px]">
                <div>Model</div>
                {sotaScore && (
                  <div className="text-xs font-normal mt-1">
                    <span className="font-semibold text-foreground">
                      SOTA: {formatSotaModelName(sotaScore.sota_model_name)}
                    </span>
                    {" "}
                    <span className="font-mono">({sotaScore.sota_score.toFixed(1)})</span>
                  </div>
                )}
              </th>
              {CONCURRENCY_TIERS.map((tier) => (
                <th
                  key={tier.key}
                  className="text-center p-3 text-sm font-medium text-muted-foreground"
                >
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="cursor-help">
                        <div>{tier.label}</div>
                        <div className="text-xs font-normal">
                          {tier.midpoint} streams
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      {tier.description}
                    </TooltipContent>
                  </Tooltip>
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
                  {/* Model info column */}
                  <td className="p-3">
                    <ModelInfo cell={cell0} rowIdx={rowIdx} />
                  </td>

                  {/* Concurrency tier columns */}
                  {row.map((cell, colIdx) => {
                    const heatmapVal = useThroughput ? cell.decodeThroughputTokS : cell.costPerStreamPerMonth;
                    const heatmap = getCellHeatmapClass(
                      heatmapVal, colMin[colIdx], colMax[colIdx], cell.exceedsCapacity, useThroughput,
                    );

                    return (
                      <td
                        key={colIdx}
                        className={`p-3 text-center align-top ${
                          cell.exceedsCapacity ? "bg-muted/20" : heatmap
                        }`}
                      >
                        <CellContent cell={cell} persona={persona} currencySymbol={currencySymbol} />
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </TooltipProvider>
  );
}
