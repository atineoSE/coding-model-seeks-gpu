"use client";

import { useMemo, useState } from "react";
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
  selfHostingCostPerRequest,
  selfHostingFloorCostPerRequest,
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


const OPEN_MODEL_COLORS = ["#22c55e", "#14b8a6", "#84cc16"];

function niceYTicks(maxY: number, extras: number[]): number[] {
  if (!(maxY > 0)) return [0];
  const rawStep = maxY / 4;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const step = Math.ceil(rawStep / magnitude) * magnitude;
  const ticks: number[] = [];
  // Keep raw values (no rounding) — per-request costs are sub-dollar.
  for (let v = 0; v <= maxY * 1.01; v += step) ticks.push(v);
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
  currencySymbol = "$",
}: ApiHostingChartProps) {
  const [requestsPerConversation, setRequestsPerConversation] = useState(DEFAULT_REQUESTS);
  const [cacheHitRate, setCacheHitRate] = useState(DEFAULT_CACHE_HIT_RATE);

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

  const {
    chartData,
    avgCosts,
    crossovers,
    floorPerReq,
    selfConfig,
    minX,
    fixedMaxX,
    fixedMaxY,
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

    // API price per request (flat — metered) for each closed model.
    const avgCosts = closedPricing.map((entry, i) =>
      computeAvgCostPerRequest(entry, configs[i]),
    );

    const selfConfig = openModels[0]
      ? computeSelfHostingCostForConfig(openModels[0], gpuConfig, gpus, settings, memoryUtilization)
      : null;

    // Self-hosting cost per request at monthly volume r: ceil(r/C) boxes'
    // fixed cost amortized over r. Falls as a box fills, floors at B/C, then
    // steps up when another box is added (sawtooth).
    const selfPerReq = (r: number): number | null =>
      selfConfig ? selfHostingCostPerRequest(r, selfConfig) : null;
    const floorPerReq = selfConfig ? selfHostingFloorCostPerRequest(selfConfig) : null;
    const B = selfConfig?.baseMonthlyCost ?? null;

    // Break-even vs each API model: the first-box crossover r* = B / price,
    // only meaningful when the box's full-utilization cost (floor) can undercut
    // that price (else self-hosting is never cheaper). `util` is how full the
    // box already is at break-even (r*/capacity) — a low % means self-hosting
    // pays off well before the box is saturated.
    const capacity = selfConfig?.maxRequestsPerMonth ?? null;
    const crossovers = B == null ? [] : closedPricing
      .map((entry, ci) => ({ entry, ci, price: avgCosts[ci] }))
      .filter(({ price }) => price > 0 && (floorPerReq == null || floorPerReq <= price))
      .map(({ entry, ci, price }) => {
        const x = B / price;
        return {
          x,
          y: price,
          closedModel: entry,
          closedIndex: ci,
          util: capacity != null && capacity > 0 ? (x / capacity) * 100 : null,
        };
      })
      .sort((a, b) => a.x - b.x);

    // Frame the x axis so the last (largest) break-even sits at 80% of the width.
    const maxCrossX = crossovers.reduce((m, c) => Math.max(m, c.x), 0);
    const minX = Math.max(100, Math.round(maxCrossX * 0.01));
    const fixedMaxX = maxCrossX > 0
      ? minX + (maxCrossX - minX) / 0.8
      : Math.max((selfConfig?.maxRequestsPerMonth ?? 0) * 2, 10_000);

    const STEPS = 240;
    const chartData = Array.from({ length: STEPS + 1 }, (_, i) => {
      const x = minX + (fixedMaxX - minX) * (i / STEPS);
      const point: Record<string, number> = { x };
      for (let ci = 0; ci < closedPricing.length; ci++) point[closedPricing[ci].lab] = avgCosts[ci];
      const sp = selfPerReq(x);
      if (sp != null) point["selfHosting"] = sp;
      return point;
    });

    // Y range ($/request). Frames the API prices + the self-hosting floor,
    // where the break-evens live; the rising low-volume tail is clipped.
    const maxApi = Math.max(0, ...avgCosts.filter((c) => c > 0));
    const fixedMaxY = (Math.max(maxApi, floorPerReq ?? 0) || maxApi || 1) * 1.6;

    return { chartData, avgCosts, crossovers, floorPerReq, selfConfig, minX, fixedMaxX, fixedMaxY };
  }, [
    closedPricing,
    openModels,
    gpuConfig,
    gpus,
    memoryUtilization,
    settings,
    requestsPerConversation,
    cacheHitRate,
  ]);

  // Closed-model labels sit on their flat line, near the right edge, nudged
  // apart so they don't overlap when prices are close.
  const closedLabelPositions = useMemo(() => {
    const minGap = fixedMaxY * 0.06;
    const entries = closedPricing.map((_, ci) => ({ ci, x: fixedMaxX * 0.62, y: avgCosts[ci] }));
    entries.sort((a, b) => b.y - a.y);
    for (let i = 1; i < entries.length; i++) {
      if (entries[i - 1].y - entries[i].y < minGap) {
        entries[i].y = Math.max(entries[i - 1].y - minGap, minGap);
      }
    }
    return new Map(entries.map(({ ci, x, y }) => [ci, { x, y }]));
  }, [closedPricing, fixedMaxX, fixedMaxY, avgCosts]);

  const selfModelName = openModels[0] ? formatModelName(openModels[0].model_name) : "";

  const yAxisTicks = useMemo(
    () => niceYTicks(fixedMaxY, floorPerReq != null && floorPerReq > 0 ? [floorPerReq] : []),
    [fixedMaxY, floorPerReq],
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

  // Per-request costs are sub-cent, so show them per MILLION requests. Takes a
  // per-request value and formats it as the cost for 1,000,000 requests.
  function formatPerMillion(perRequest: number) {
    const v = perRequest * 1_000_000;
    if (v >= 1_000_000) return `${currencySymbol}${(v / 1_000_000).toFixed(2)}M`;
    if (v >= 1_000) return `${currencySymbol}${(v / 1_000).toFixed(0)}k`;
    if (v >= 1) return `${currencySymbol}${v.toFixed(0)}`;
    if (v > 0) return `${currencySymbol}${v.toFixed(2)}`;
    return `${currencySymbol}0`;
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
          Cost per million requests at a given monthly volume. Solid lines = API (flat, metered per request); the dashed curve = self-hosting, amortizing the box&apos;s fixed cost over its throughput — it falls as the box fills, floors at the full-utilization cost, and steps up when another box is added. Each dot marks where self-hosting undercuts an API model, labelled with the box utilization (requests served ÷ monthly capacity) needed to break even — a low % means self-hosting pays off well before the box is full.
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
        </div>

        {selfConfig && (
          <p className="text-sm text-muted-foreground">
            Self-hosting on{" "}
            <span className="font-medium text-foreground">
              {gpuConfig.gpuCount}× {gpuConfig.gpuName}
            </span>{" · "}
            <span className="font-medium text-foreground">
              {currencySymbol}{Math.round(selfConfig.baseMonthlyCost).toLocaleString("en-US")}/mo
            </span>
          </p>
        )}

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
              domain={[0, fixedMaxY]}
              allowDataOverflow
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              ticks={yAxisTicks}
              tickFormatter={formatPerMillion}
              label={{
                value: "Cost per 1M requests",
                angle: -90,
                position: "insideLeft",
                offset: 4,
                fontSize: 12,
                fill: "var(--muted-foreground)",
              }}
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
                            {formatPerMillion(p.value as number)} / 1M req
                          </span>
                        </div>
                      ))}
                  </div>
                );
              }}
            />

            {/* Self-hosting full-utilization floor (best-case cost/request). */}
            {floorPerReq != null && floorPerReq > 0 && (
              <ReferenceLine
                y={floorPerReq}
                stroke={OPEN_MODEL_COLORS[0] ?? "#22c55e"}
                strokeWidth={1}
                strokeDasharray="2 3"
                label={{
                  value: `full-utilization ${formatPerMillion(floorPerReq)} / 1M req`,
                  position: "insideTopRight",
                  fontSize: 10,
                  fill: OPEN_MODEL_COLORS[0] ?? "#22c55e",
                }}
              />
            )}

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

            {selfConfig && (
              <Line
                type="linear"
                dataKey="selfHosting"
                stroke={OPEN_MODEL_COLORS[0] ?? "#22c55e"}
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={false}
                name={selfModelName}
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

            {crossovers.map((ix) => (
              <ReferenceLine
                key={`vline-${ix.closedModel.lab}`}
                x={ix.x}
                stroke={CLOSED_MODEL_COLORS[ix.closedModel.lab] ?? "#888"}
                strokeWidth={1}
                strokeDasharray="4 3"
                label={{
                  value: formatRequests(ix.x),
                  position: "insideBottomLeft",
                  fontSize: 10,
                  fill: CLOSED_MODEL_COLORS[ix.closedModel.lab] ?? "#888",
                  offset: 4,
                }}
              />
            ))}

            {crossovers.map((ix) => {
              const util = ix.util;
              return (
                <ReferenceDot
                  key={`dot-${ix.closedModel.lab}`}
                  x={ix.x}
                  y={ix.y}
                  r={5}
                  fill="white"
                  stroke={CLOSED_MODEL_COLORS[ix.closedModel.lab] ?? "#888"}
                  strokeWidth={2}
                  label={util == null ? undefined : (props: { viewBox?: { x?: number; y?: number } }) => (
                    <text
                      x={props.viewBox?.x ?? 0}
                      y={(props.viewBox?.y ?? 0) - 10}
                      textAnchor="middle"
                      fontSize={10}
                      fontWeight={600}
                      fill={CLOSED_MODEL_COLORS[ix.closedModel.lab] ?? "#888"}
                    >
                      {util >= 10 ? Math.round(util) : util.toFixed(1)}% used
                    </text>
                  )}
                />
              );
            })}
          </LineChart>
        </ChartContainer>

        {openModels.length > 0 && !selfConfig && (
          <p className="text-sm text-muted-foreground">
            No GPU offering found for the selected GPU configuration.
          </p>
        )}

        {selfConfig && crossovers.length === 0 && (
          <p className="text-sm text-muted-foreground">
            At this box&apos;s full-utilization cost{floorPerReq != null ? ` (${formatPerMillion(floorPerReq)} / 1M req)` : ""}, self-hosting {selfModelName} stays pricier than every API option shown — no break-even.
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
