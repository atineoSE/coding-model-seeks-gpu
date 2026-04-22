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
import type { ApiPricingEntry, Model, GpuOffering, AdvancedSettings } from "@/types";
import {
  computeAvgCostPerTurn,
  computeSelfHostingMonthlyCost,
  findIntersection,
  getProviderCacheTtls,
  type CostConfig,
} from "@/lib/api-hosting-cost";

const TURNS_OPTIONS = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500];
const CACHE_HIT_RATE_OPTIONS = [0.8, 0.85, 0.9, 0.95, 0.99];
const DEFAULT_TURNS = 150;
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

interface ApiHostingChartProps {
  closedPricing: ApiPricingEntry[];
  openModels: Model[];
  gpus: GpuOffering[];
  settings: AdvancedSettings;
  currencySymbol?: string;
}

export function ApiHostingChart({
  closedPricing,
  openModels,
  gpus,
  settings,
  currencySymbol = "$",
}: ApiHostingChartProps) {
  const [turnsPerConversation, setTurnsPerConversation] = useState(DEFAULT_TURNS);
  const [cacheHitRate, setCacheHitRate] = useState(DEFAULT_CACHE_HIT_RATE);

  const chartConfig = useMemo(() => {
    const cfg: Record<string, { label: string; color: string }> = {};
    for (const entry of closedPricing) {
      cfg[entry.lab] = {
        label: entry.model_name,
        color: CLOSED_MODEL_COLORS[entry.lab] ?? "#888",
      };
    }
    for (let i = 0; i < openModels.length; i++) {
      cfg[`open_${i}`] = { label: openModels[i].model_name, color: OPEN_MODEL_COLORS[i] ?? "#888" };
    }
    return cfg as ChartConfig;
  }, [closedPricing, openModels]);

  const { chartData, openCosts, intersections, avgCosts, fixedMaxX, fixedMaxY, minX } = useMemo(() => {
    const configs: CostConfig[] = closedPricing.map((entry) => ({
      turnsPerConversation,
      cacheHitRate,
      cacheTtlMin: getProviderCacheTtls(entry).length > 0 ? Math.min(...getProviderCacheTtls(entry)) : null,
      avgInputTokens: settings.avgInputTokens,
      avgOutputTokens: settings.avgOutputTokens,
    }));

    const avgCosts = closedPricing.map((entry, i) =>
      computeAvgCostPerTurn(entry, configs[i]),
    );

    const openCosts = openModels.map((model) => ({
      model,
      cost: computeSelfHostingMonthlyCost(model, gpus, settings),
    }));

    const intersections: {
      x: number;
      y: number;
      closedModel: ApiPricingEntry;
      openModel: Model;
      closedIndex: number;
      openIndex: number;
    }[] = [];

    for (let ci = 0; ci < closedPricing.length; ci++) {
      for (let oi = 0; oi < openCosts.length; oi++) {
        const flatCost = openCosts[oi].cost;
        if (flatCost == null) continue;
        const ix = findIntersection(avgCosts[ci], flatCost);
        if (ix == null) continue;
        intersections.push({
          x: ix,
          y: flatCost,
          closedModel: closedPricing[ci],
          openModel: openCosts[oi].model,
          closedIndex: ci,
          openIndex: oi,
        });
      }
    }

    const maxIntersection = intersections.reduce((max, ix) => Math.max(max, ix.x), 0);
    const fixedMaxX = Math.max(maxIntersection * 1.05, 10_000);
    const minX = 100;
    const top3Intersections = intersections.filter((ix) => ix.openIndex < 3);
    const maxIntersectionY = top3Intersections.reduce((max, ix) => Math.max(max, ix.y), 0);
    const fallbackMaxY = openCosts.slice(0, 3).reduce((max, { cost }) => (cost != null ? Math.min(max, cost) : max), Infinity);
    const fixedMaxY = (maxIntersectionY > 0 ? maxIntersectionY : fallbackMaxY) * 1.2;

    // Log-spaced points so curves render smoothly on the log X axis
    const STEPS = 200;
    const chartData = Array.from({ length: STEPS + 1 }, (_, i) => {
      const x = minX * Math.pow(fixedMaxX / minX, i / STEPS);
      const point: Record<string, number> = { x };
      for (let ci = 0; ci < closedPricing.length; ci++) {
        point[closedPricing[ci].lab] = x * avgCosts[ci];
      }
      return point;
    });

    return { chartData, openCosts, intersections, avgCosts, fixedMaxX, fixedMaxY, minX };
  }, [closedPricing, openModels, gpus, settings, turnsPerConversation, cacheHitRate]);

  const sortedIntersections = useMemo(
    () => [...intersections].sort((a, b) => a.x - b.x),
    [intersections],
  );

  function formatTurns(n: number) {
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
          Monthly cost at a given turns/month volume. Solid lines = API; dashed lines = self-hosting.
          Dots mark the break-even point per pair.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-x-6 gap-y-3">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-muted-foreground whitespace-nowrap">
              Turns / conversation
            </label>
            <Select
              value={String(turnsPerConversation)}
              onValueChange={(v) => setTurnsPerConversation(Number(v))}
            >
              <SelectTrigger className="w-[80px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TURNS_OPTIONS.map((t) => (
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
              scale="log"
              domain={[minX, fixedMaxX]}
              allowDataOverflow
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={formatTurns}
              label={{
                value: "Agent turns per month",
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
                    <p className="font-medium">{formatTurns(x)} turns/mo</p>
                    {[...payload].sort((a, b) => (b.value as number) - (a.value as number)).map((p) => (
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
                name={entry.model_name}
                allowDataOverflow
              />
            ))}

            {closedPricing.map((entry, ci) => (
              <ReferenceDot
                key={`label-${entry.lab}`}
                x={fixedMaxX}
                y={fixedMaxX * avgCosts[ci]}
                r={0}
                label={{
                  value: entry.model_name,
                  position: "insideTopRight",
                  fontSize: 10,
                  fill: CLOSED_MODEL_COLORS[entry.lab] ?? "#888",
                }}
              />
            ))}

            {openCosts.map(({ model, cost }, i) =>
              cost != null ? (
                <ReferenceLine
                  key={model.model_name}
                  y={cost}
                  stroke={OPEN_MODEL_COLORS[i] ?? "#888"}
                  strokeWidth={1.5}
                  strokeDasharray="6 3"
                  label={{
                    value: model.model_name,
                    position: "insideTopRight",
                    fontSize: 10,
                    fill: OPEN_MODEL_COLORS[i] ?? "#888",
                  }}
                />
              ) : null,
            )}

            {sortedIntersections.map((ix) => (
              <ReferenceLine
                key={`vline-${ix.closedModel.lab}-${ix.openModel.model_name}`}
                x={ix.x}
                stroke={OPEN_MODEL_COLORS[ix.openIndex] ?? "#888"}
                strokeWidth={1}
                strokeDasharray="4 3"
                label={{
                  value: formatTurns(ix.x),
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

        {intersections.length > 0 && (
          <div className="text-sm space-y-2">
            <p className="font-medium text-muted-foreground">Break-even points</p>
            <div className="space-y-1">
              {sortedIntersections.map((ix) => (
                <div
                  key={`${ix.closedModel.lab}-${ix.openModel.model_name}`}
                  className="flex items-center gap-2"
                >
                  <span
                    className="inline-block h-3 w-3 rounded-full flex-shrink-0 border-2"
                    style={{
                      backgroundColor: "white",
                      borderColor: OPEN_MODEL_COLORS[ix.openIndex] ?? "#888",
                    }}
                  />
                  <span className="text-muted-foreground">
                    {ix.closedModel.model_name} / {ix.openModel.model_name}:
                  </span>
                  <span className="font-medium tabular-nums">
                    {ix.x.toLocaleString("en-US", { maximumFractionDigits: 0 })} turns/mo
                  </span>
                  <span className="text-muted-foreground text-xs">
                    ({formatCost(ix.y)}/mo self-hosting)
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {openModels.length > 0 && openCosts.every(({ cost }) => cost == null) && (
          <p className="text-sm text-muted-foreground">
            No GPU setup data available for the selected open-weight models.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
