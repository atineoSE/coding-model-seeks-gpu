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

  const { chartData, openCosts, intersections, maxX } = useMemo(() => {
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

    const maxIntersectionX =
      intersections.length > 0 ? Math.max(...intersections.map((p) => p.x)) : 0;
    const maxX = Math.max(maxIntersectionX * 1.5, 10_000);

    const STEPS = 100;
    const chartData = Array.from({ length: STEPS + 1 }, (_, i) => {
      const x = (maxX / STEPS) * i;
      const point: Record<string, number> = { x };
      for (let ci = 0; ci < closedPricing.length; ci++) {
        point[closedPricing[ci].lab] = x * avgCosts[ci];
      }
      return point;
    });

    return { chartData, openCosts, intersections, maxX };
  }, [closedPricing, openModels, gpus, settings, turnsPerConversation, cacheHitRate]);

  const sortedIntersections = useMemo(
    () => [...intersections].sort((a, b) => a.x - b.x),
    [intersections],
  );

  // Fall back to legend table when labels would crowd the chart
  const overlapThreshold = maxX * 0.05;
  const hasOverlaps = intersections.some((a, i) =>
    intersections.some((b, j) => i < j && Math.abs(a.x - b.x) < overlapThreshold),
  );
  const showInlineLabels = !hasOverlaps && intersections.length <= 4;

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
              domain={[0, maxX]}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={formatTurns}
              label={{
                value: "Turns per month",
                position: "insideBottom",
                offset: -16,
                fontSize: 12,
                fill: "var(--muted-foreground)",
              }}
            />
            <YAxis
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
                    {payload.map((p) => (
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

            {sortedIntersections.map((ix, i) => (
              <ReferenceDot
                key={`${ix.closedModel.lab}-${ix.openModel.model_name}`}
                x={ix.x}
                y={ix.y}
                r={5}
                fill="white"
                stroke={OPEN_MODEL_COLORS[ix.openIndex] ?? "#888"}
                strokeWidth={2}
                label={
                  showInlineLabels
                    ? {
                        value: `${formatTurns(ix.x)}/mo`,
                        position: i % 2 === 0 ? "top" : "bottom",
                        fontSize: 10,
                        fill: "var(--foreground)",
                        offset: 8,
                      }
                    : undefined
                }
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
