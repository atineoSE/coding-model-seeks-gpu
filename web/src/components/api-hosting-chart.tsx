"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  ReferenceDot,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ApiPricingEntry, Model, GpuOffering, AdvancedSettings, BenchmarkScore, PresetGpuConfig } from "@/types";
import {
  computeAvgCostPerRequest,
  computeSelfHostingCostForConfig,
  getProviderCacheTtls,
  type CostConfig,
} from "@/lib/api-hosting-cost";
import { formatModelName } from "@/lib/utils";

const REQUESTS_OPTIONS = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];
const CACHE_HIT_RATE_OPTIONS = [0.8, 0.85, 0.9, 0.95, 0.99];
const DEFAULT_REQUESTS = 150;
const DEFAULT_CACHE_HIT_RATE = 0.9;

const CLOSED_MODEL_COLORS: Record<string, string> = {
  anthropic: "#f59e0b",
  openai: "#3b82f6",
  google: "#8b5cf6",
};

const LAB_DISPLAY_NAMES: Record<string, string> = {
  anthropic: "Anthropic",
  openai: "OpenAI",
  google: "Google",
};

function labToDisplayName(lab: string): string {
  return LAB_DISPLAY_NAMES[lab] ?? lab.charAt(0).toUpperCase() + lab.slice(1);
}

const OPEN_MODEL_COLORS = ["#22c55e", "#14b8a6", "#84cc16"];

function niceYTicks(maxY: number, extras: number[]): number[] {
  const rawStep = maxY / 4;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const step = Math.ceil(rawStep / magnitude) * magnitude;
  const ticks: number[] = [];
  for (let v = 0; v <= maxY * 1.01; v += step) ticks.push(Math.round(v));
  const validExtras = extras.filter((e) => e > 0);
  if (validExtras.length === 0) return ticks;
  const filtered = ticks.filter(
    (t) => !validExtras.some((e) => Math.abs(t - e) <= e * 0.08),
  );
  return [...filtered, ...validExtras].sort((a, b) => a - b);
}

interface ApiHostingChartProps {
  closedPricing: ApiPricingEntry[];
  availableModels: Array<{ model: Model; sotaPercent: number | null }>;
  gpuConfig: PresetGpuConfig;
  gpus: GpuOffering[];
  memoryUtilization: number;
  settings: AdvancedSettings;
  benchmarks: BenchmarkScore[];
  benchmarkCategory: string;
  currencySymbol?: string;
}

export function ApiHostingChart({
  closedPricing,
  availableModels,
  gpuConfig,
  gpus,
  memoryUtilization,
  settings,
  benchmarks,
  benchmarkCategory,
  currencySymbol = "$",
}: ApiHostingChartProps) {
  const [requestsPerConversation, setRequestsPerConversation] = useState(DEFAULT_REQUESTS);
  const [cacheHitRate, setCacheHitRate] = useState(DEFAULT_CACHE_HIT_RATE);
  const [yZoomed, setYZoomed] = useState(false);

  const openModels = useMemo(
    () => (availableModels[0] ? [availableModels[0].model] : []),
    [availableModels],
  );

  const chartConfig = useMemo(() => {
    const cfg: Record<string, { label: string; color: string }> = {};
    for (const entry of closedPricing) {
      cfg[entry.lab] = {
        label: entry.model_name,
        color: CLOSED_MODEL_COLORS[entry.lab] ?? "#888",
      };
    }
    if (openModels.length > 0) {
      cfg["selfHosting"] = {
        label: openModels[0].model_name,
        color: OPEN_MODEL_COLORS[0] ?? "#22c55e",
      };
    }
    return cfg as ChartConfig;
  }, [closedPricing, openModels]);

  const benchmarkScoreMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const b of benchmarks) {
      if (b.benchmark_name === benchmarkCategory && b.score !== null) {
        const existing = map.get(b.model_name);
        if (existing === undefined || b.score > existing) map.set(b.model_name, b.score);
      }
    }
    return map;
  }, [benchmarks, benchmarkCategory]);

  const benchmarkDisplayName = useMemo(
    () =>
      benchmarks.find((b) => b.benchmark_name === benchmarkCategory)?.benchmark_display_name ??
      benchmarkCategory,
    [benchmarks, benchmarkCategory],
  );

  const {
    chartData,
    openCosts,
    intersections,
    avgCosts,
    fixedMaxX,
    fixedMaxY,
    fullMaxY,
    minX,
  } = useMemo(() => {
    const configs: CostConfig[] = closedPricing.map((entry) => ({
      requestsPerConversation,
      cacheHitRate,
      cacheTtlMin:
        getProviderCacheTtls(entry).length > 0
          ? Math.min(...getProviderCacheTtls(entry))
          : null,
      avgInputTokens: settings.avgInputTokens,
      avgOutputTokens: settings.avgOutputTokens,
    }));

    const avgCosts = closedPricing.map((entry, i) =>
      computeAvgCostPerRequest(entry, configs[i]),
    );

    const openCosts = openModels.map((model) => ({
      model,
      costConfig: computeSelfHostingCostForConfig(
        model,
        gpuConfig,
        gpus,
        settings,
        memoryUtilization,
      ),
    }));

    const allIntersections: {
      x: number;
      y: number;
      closedModel: ApiPricingEntry;
      openModel: Model;
      closedIndex: number;
      openIndex: number;
      performanceNote: string | null;
    }[] = [];

    for (let ci = 0; ci < closedPricing.length; ci++) {
      const avgCostPerRequest = avgCosts[ci];
      if (avgCostPerRequest <= 0) continue;

      for (let oi = 0; oi < openCosts.length; oi++) {
        const { model, costConfig } = openCosts[oi];
        if (!costConfig) continue;

        const { baseMonthlyCost, maxRequestsPerMonth } = costConfig;

        // Skip if self-hosting is always more expensive than API at full capacity
        if (maxRequestsPerMonth !== null && baseMonthlyCost > maxRequestsPerMonth * avgCostPerRequest) continue;

        const closedScore = benchmarkScoreMap.get(closedPricing[ci].model_name) ?? null;
        const openScore = benchmarkScoreMap.get(model.model_name) ?? null;
        let performanceNote: string | null = null;
        if (closedScore !== null && openScore !== null && openScore > 0 && closedScore > 0) {
          const [source, reference, pct] =
            openScore <= closedScore
              ? [model.model_name, closedPricing[ci].model_name, (openScore / closedScore) * 100]
              : [closedPricing[ci].model_name, model.model_name, (closedScore / openScore) * 100];
          performanceNote = `${source} has ${pct.toFixed(1)}% performance of ${reference} (${benchmarkDisplayName})`;
        }

        allIntersections.push({
          x: baseMonthlyCost / avgCostPerRequest,
          y: baseMonthlyCost,
          closedModel: closedPricing[ci],
          openModel: model,
          closedIndex: ci,
          openIndex: oi,
          performanceNote,
        });
      }
    }

    const maxIntersectionX = allIntersections.reduce((max, ix) => Math.max(max, ix.x), 0);
    const fixedMaxX = Math.max(maxIntersectionX * 1.20, 10_000);
    const minX = 100;

    const intersections = allIntersections.filter((ix) => ix.x <= fixedMaxX);

    const openCostConfig = openCosts[0]?.costConfig ?? null;
    const selfHostingCost = openCostConfig?.baseMonthlyCost ?? 0;

    const maxIntersectionY = intersections
      .filter((ix) => ix.openIndex < 3)
      .reduce((max, ix) => Math.max(max, ix.y), 0);
    const fixedMaxY = Math.max(
      maxIntersectionY > 0 ? maxIntersectionY * 1.2 : 0,
      selfHostingCost * 1.1,
    ) || 10_000;

    const fullMaxY = Math.max(...avgCosts.filter(c => c > 0), 0) * fixedMaxX * 1.05 || fixedMaxY;

    const STEPS = 200;
    const chartData = Array.from({ length: STEPS + 1 }, (_, i) => {
      const x = minX + (fixedMaxX - minX) * (i / STEPS);
      const point: Record<string, number> = { x };
      for (let ci = 0; ci < closedPricing.length; ci++) {
        point[closedPricing[ci].lab] = x * avgCosts[ci];
      }
      if (openCostConfig) {
        point["selfHosting"] = openCostConfig.baseMonthlyCost;
      }
      return point;
    });

    return { chartData, openCosts, intersections, avgCosts, fixedMaxX, fixedMaxY, fullMaxY, minX };
  }, [
    closedPricing,
    openModels,
    gpuConfig,
    gpus,
    memoryUtilization,
    settings,
    requestsPerConversation,
    cacheHitRate,
    benchmarkScoreMap,
    benchmarkDisplayName,
  ]);

  const activeMaxY = yZoomed ? fullMaxY : fixedMaxY;

  const animFrameRef = useRef<number | null>(null);
  const fromMaxYRef = useRef(activeMaxY);
  const [renderedMaxY, setRenderedMaxY] = useState(activeMaxY);

  useEffect(() => {
    const from = fromMaxYRef.current;
    const to = activeMaxY;
    if (from === to) return;

    const startTime = performance.now();
    const DURATION = 350;

    if (animFrameRef.current != null) cancelAnimationFrame(animFrameRef.current);

    function tick(now: number) {
      const t = Math.min((now - startTime) / DURATION, 1);
      const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      const val = from + (to - from) * eased;
      fromMaxYRef.current = val;
      setRenderedMaxY(val);
      if (t < 1) {
        animFrameRef.current = requestAnimationFrame(tick);
      } else {
        fromMaxYRef.current = to;
      }
    }

    animFrameRef.current = requestAnimationFrame(tick);
    return () => { if (animFrameRef.current != null) cancelAnimationFrame(animFrameRef.current); };
  }, [activeMaxY]);

  // Place each closed-model label near the last visible point on that curve.
  const closedLabelPositions = useMemo(() => {
    const ANCHOR = 0.65;
    const minGap = renderedMaxY * 0.06;

    const entries = closedPricing.map((_, ci) => {
      const exitX = avgCosts[ci] > 0 ? renderedMaxY / avgCosts[ci] : fixedMaxX;
      const anchorX = Math.min(fixedMaxX, exitX) * ANCHOR;
      const anchorY = anchorX * avgCosts[ci];
      return { ci, x: anchorX, y: anchorY };
    });

    entries.sort((a, b) => b.y - a.y);
    for (let i = 1; i < entries.length; i++) {
      if (entries[i - 1].y - entries[i].y < minGap) {
        entries[i].y = Math.max(entries[i - 1].y - minGap, minGap);
      }
    }

    return new Map(entries.map(({ ci, x, y }) => [ci, { x, y }]));
  }, [closedPricing, fixedMaxX, renderedMaxY, avgCosts]);

  const selfHostingCost = openCosts[0]?.costConfig?.baseMonthlyCost ?? 0;

  const yAxisTicks = useMemo(
    () => niceYTicks(renderedMaxY, selfHostingCost > 0 ? [selfHostingCost] : []),
    [renderedMaxY, selfHostingCost],
  );

  const sortedIntersections = useMemo(
    () => [...intersections].sort((a, b) => a.x - b.x),
    [intersections],
  );

  const modelCapacitiesReqH = useMemo(() =>
    availableModels.map(({ model }) => {
      const cfg = computeSelfHostingCostForConfig(model, gpuConfig, gpus, settings, memoryUtilization);
      return cfg?.maxRequestsPerMonth != null ? Math.round(cfg.maxRequestsPerMonth / 720) : null;
    }),
    [availableModels, gpuConfig, gpus, settings, memoryUtilization],
  );

  function formatRequests(n: number) {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
    return String(Math.round(n));
  }

  function formatCost(n: number) {
    return `${currencySymbol}${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  }

  if (closedPricing.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>API vs. Self-Hosting Cost</CardTitle>
          <CardDescription>No API pricing data available.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>API vs. Self-Hosting Cost</CardTitle>
        <CardDescription>
          Monthly cost at a given requests/month volume. Solid lines = API; dashed line = self-hosting (one instance). Dots mark the break-even point per pair.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-muted-foreground whitespace-nowrap">
              Requests / conversation
            </label>
            <Select
              value={String(requestsPerConversation)}
              onValueChange={(v) => setRequestsPerConversation(Number(v))}
            >
              <SelectTrigger className="w-[80px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {REQUESTS_OPTIONS.map((t) => (
                  <SelectItem key={t} value={String(t)}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-muted-foreground whitespace-nowrap">
              Cache hit rate
            </label>
            <Select
              value={String(cacheHitRate)}
              onValueChange={(v) => setCacheHitRate(Number(v))}
            >
              <SelectTrigger className="w-[90px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CACHE_HIT_RATE_OPTIONS.map((r) => (
                  <SelectItem key={r} value={String(r)}>
                    {Math.round(r * 100)}%
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <button
            onClick={() => setYZoomed((z) => !z)}
            className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            title={yZoomed ? "Zoom in" : "Zoom out"}
          >
            {yZoomed ? (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="6" cy="6" r="4" />
                <path d="M9 9l3 3M4 6h4M6 4v4" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="6" cy="6" r="4" />
                <path d="M9 9l3 3M4 6h4" />
              </svg>
            )}
            {yZoomed ? "Zoom in" : "Zoom out"}
          </button>
        </div>

        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-[300px] sm:h-[380px] w-full"
        >
          <LineChart
            data={chartData}
            margin={{ top: 24, right: 24, bottom: 32, left: 16 }}
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis
              dataKey="x"
              type="number"
              domain={[minX, fixedMaxX]}
              allowDataOverflow
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={formatRequests}
              label={{
                value: "Requests per month",
                position: "insideBottom",
                offset: -16,
                fontSize: 12,
                fill: "var(--muted-foreground)",
              }}
            />
            <YAxis
              domain={[0, renderedMaxY]}
              allowDataOverflow
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              ticks={yAxisTicks}
              tickFormatter={(v: number) =>
                `${currencySymbol}${v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)}`
              }
            />
            <ChartTooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const x = payload[0]?.payload?.x as number | undefined;
                if (x == null) return null;
                return (
                  <div className="rounded-lg border bg-background p-3 shadow-sm text-sm space-y-1">
                    <p className="font-medium">{formatRequests(x)} requests/mo</p>
                    {[...payload]
                      .sort((a, b) => (a.value as number) - (b.value as number))
                      .map((p) => (
                        <div key={String(p.dataKey)} className="flex items-center gap-2">
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full flex-shrink-0"
                            style={{ backgroundColor: p.color }}
                          />
                          <span className="text-muted-foreground">{p.name}:</span>
                          <span className="font-medium">
                            {formatCost(p.value as number)}/mo
                          </span>
                        </div>
                      ))}
                  </div>
                );
              }}
            />

            {closedPricing.map((entry) => (
              <Line
                key={entry.lab}
                type="linear"
                dataKey={entry.lab}
                stroke={CLOSED_MODEL_COLORS[entry.lab] ?? "#888"}
                strokeWidth={2}
                dot={false}
                name={formatModelName(entry.model_name)}
              />
            ))}

            {openCosts[0]?.costConfig && (
              <Line
                type="linear"
                dataKey="selfHosting"
                stroke={OPEN_MODEL_COLORS[0] ?? "#22c55e"}
                strokeWidth={1.5}
                strokeDasharray="6 3"
                dot={false}
                name={formatModelName(openCosts[0].model.model_name)}
              />
            )}

            {closedPricing.map((entry, ci) => {
              const pos = closedLabelPositions.get(ci);
              if (!pos) return null;
              return (
                <ReferenceDot
                  key={`label-${entry.lab}`}
                  x={pos.x}
                  y={pos.y}
                  r={0}
                  label={(props: { viewBox?: { x?: number; y?: number } }) => (
                    <text
                      x={props.viewBox?.x ?? 0}
                      y={(props.viewBox?.y ?? 0) - 6}
                      textAnchor="middle"
                      fontSize={10}
                      fill={CLOSED_MODEL_COLORS[entry.lab] ?? "#888"}
                    >
                      {formatModelName(entry.model_name)}
                    </text>
                  )}
                />
              );
            })}

            {openCosts[0]?.costConfig && (
              <ReferenceDot
                x={(minX + fixedMaxX) / 2}
                y={openCosts[0].costConfig.baseMonthlyCost}
                r={0}
                label={(props: { viewBox?: { x?: number; y?: number } }) => (
                  <text
                    x={props.viewBox?.x ?? 0}
                    y={(props.viewBox?.y ?? 0) - 6}
                    textAnchor="middle"
                    fontSize={10}
                    fill={OPEN_MODEL_COLORS[0] ?? "#22c55e"}
                  >
                    {formatModelName(openCosts[0].model.model_name)}
                  </text>
                )}
              />
            )}

            {sortedIntersections.map((ix) => (
              <ReferenceLine
                key={`vline-${ix.closedModel.lab}-${ix.openModel.model_name}`}
                x={ix.x}
                stroke={OPEN_MODEL_COLORS[ix.openIndex] ?? "#888"}
                strokeWidth={1}
                strokeDasharray="4 3"
                label={{
                  value: formatRequests(ix.x),
                  position: "insideBottomLeft",
                  fontSize: 10,
                  fill: OPEN_MODEL_COLORS[ix.openIndex] ?? "#888",
                  offset: 4,
                }}
              />
            ))}

            {sortedIntersections.map((ix) => (
              <ReferenceDot
                key={`dot-${ix.closedModel.lab}-${ix.openModel.model_name}`}
                x={ix.x}
                y={ix.y}
                r={5}
                fill="white"
                stroke={OPEN_MODEL_COLORS[ix.openIndex] ?? "#888"}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ChartContainer>

{openModels.length > 0 && openCosts.every(({ costConfig }) => costConfig == null) && (
          <p className="text-sm text-muted-foreground">
            No GPU offering found for the selected GPU configuration.
          </p>
        )}

        {availableModels.length > 0 && (
          <div className="text-sm space-y-1.5 pt-1 border-t">
            <p className="font-medium text-muted-foreground">Models served by this GPU config</p>
            <div className="flex flex-col gap-y-0.5">
              {availableModels.map((entry, i) => {
                const reqH = modelCapacitiesReqH[i] ?? null;
                const reqHFormatted = reqH != null
                  ? (reqH >= 10_000 ? `${Math.round(reqH / 1000)}k` : reqH >= 1_000 ? `${(reqH / 1000).toFixed(1)}k` : String(reqH))
                  : null;
                const detail = [
                  entry.sotaPercent !== null ? `${Math.round(entry.sotaPercent)}% of SOTA` : null,
                  reqHFormatted !== null ? `${reqHFormatted} req/h` : null,
                ].filter(Boolean).join(" at ");
                return (
                  <span key={entry.model.model_name} className="text-muted-foreground">
                    <span className={i === 0 ? "font-medium text-foreground" : ""}>
                      {formatModelName(entry.model.model_name)}
                    </span>
                    {detail && <span className="text-xs ml-1">({detail})</span>}
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
