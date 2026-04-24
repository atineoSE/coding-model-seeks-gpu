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
  selfHostingStepCost,
  getProviderCacheTtls,
  type CostConfig,
} from "@/lib/api-hosting-cost";

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
    minX,
    stepBoundaries,
    visibleStepCosts,
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
      replicaCount: number;
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

        const closedScore = benchmarkScoreMap.get(closedPricing[ci].model_name) ?? null;
        const openScore = benchmarkScoreMap.get(model.model_name) ?? null;
        let performanceNote: string | null = null;
        if (
          closedScore !== null &&
          openScore !== null &&
          openScore > 0 &&
          closedScore > 0
        ) {
          const [source, reference, pct] =
            openScore <= closedScore
              ? [
                  model.model_name,
                  closedPricing[ci].model_name,
                  (openScore / closedScore) * 100,
                ]
              : [
                  closedPricing[ci].model_name,
                  model.model_name,
                  (closedScore / openScore) * 100,
                ];
          performanceNote = `${source} has ${pct.toFixed(1)}% performance of ${reference} (${benchmarkDisplayName})`;
        }

        if (maxRequestsPerMonth === null) {
          // No capacity info: flat cost, single intersection
          const x = baseMonthlyCost / avgCostPerRequest;
          allIntersections.push({
            x,
            y: baseMonthlyCost,
            replicaCount: 1,
            closedModel: closedPricing[ci],
            openModel: model,
            closedIndex: ci,
            openIndex: oi,
            performanceNote,
          });
        } else {
          // Self-hosting always more expensive at full capacity — no crossover possible
          if (baseMonthlyCost > maxRequestsPerMonth * avgCostPerRequest) continue;

          // Step function: find the unique valid replica count k
          for (let k = 1; ; k++) {
            const stepCost = k * baseMonthlyCost;
            const candidateX = stepCost / avgCostPerRequest;
            if (candidateX > 10_000_000) break;
            const lowerBound = (k - 1) * maxRequestsPerMonth;
            const upperBound = k * maxRequestsPerMonth;
            if (candidateX > lowerBound && candidateX <= upperBound) {
              allIntersections.push({
                x: candidateX,
                y: stepCost,
                replicaCount: k,
                closedModel: closedPricing[ci],
                openModel: model,
                closedIndex: ci,
                openIndex: oi,
                performanceNote,
              });
              break;
            }
          }
        }
      }
    }

    const maxIntersectionX = allIntersections.reduce((max, ix) => Math.max(max, ix.x), 0);
    const fixedMaxX = Math.max(maxIntersectionX * 1.20, 10_000);
    const minX = 100;

    const intersections = allIntersections.filter((ix) => ix.x <= fixedMaxX);

    const openCostConfig = openCosts[0]?.costConfig ?? null;

    // Step boundaries at n × maxRequestsPerMonth within [minX, fixedMaxX]
    const stepBoundaries: number[] = [];
    if (openCostConfig?.maxRequestsPerMonth != null) {
      const T = openCostConfig.maxRequestsPerMonth;
      for (let n = 1; n * T <= fixedMaxX; n++) {
        const boundary = n * T;
        if (boundary >= minX) stepBoundaries.push(boundary);
      }
    }

    // Highest step cost visible in the chart range for fixedMaxY
    let maxVisibleStepCost = 0;
    if (openCostConfig?.maxRequestsPerMonth != null) {
      const T = openCostConfig.maxRequestsPerMonth;
      const maxReplicas = Math.max(1, Math.ceil(fixedMaxX / T));
      maxVisibleStepCost = maxReplicas * openCostConfig.baseMonthlyCost;
    } else if (openCostConfig) {
      maxVisibleStepCost = openCostConfig.baseMonthlyCost;
    }

    const top3Intersections = intersections.filter((ix) => ix.openIndex < 3);
    const maxIntersectionY = top3Intersections.reduce(
      (max, ix) => Math.max(max, ix.y),
      0,
    );
    const fixedMaxY =
      Math.max(
        maxIntersectionY > 0 ? maxIntersectionY * 1.2 : 0,
        maxVisibleStepCost * 1.1,
      ) || 10_000;

    // Step-level costs visible within the y range (for axis tick injection)
    const visibleStepCosts: number[] = [];
    if (openCostConfig?.maxRequestsPerMonth != null) {
      for (let n = 1; n * openCostConfig.baseMonthlyCost <= fixedMaxY * 1.1; n++) {
        visibleStepCosts.push(n * openCostConfig.baseMonthlyCost);
      }
    } else if (openCostConfig) {
      visibleStepCosts.push(openCostConfig.baseMonthlyCost);
    }

    // 200 log-spaced points plus boundary pairs for clean step transitions
    const STEPS = 200;
    const baseXPoints = Array.from(
      { length: STEPS + 1 },
      (_, i) => minX * Math.pow(fixedMaxX / minX, i / STEPS),
    );
    const extraXPoints: number[] = [];
    for (const t of stepBoundaries) {
      extraXPoints.push(t, t * 1.0001);
    }
    const allXPoints = [...baseXPoints, ...extraXPoints].sort((a, b) => a - b);

    const chartData = allXPoints.map((x) => {
      const point: Record<string, number> = { x };
      for (let ci = 0; ci < closedPricing.length; ci++) {
        point[closedPricing[ci].lab] = x * avgCosts[ci];
      }
      if (openCostConfig) {
        point["selfHosting"] = selfHostingStepCost(x, openCostConfig);
      }
      return point;
    });

    return {
      chartData,
      openCosts,
      intersections,
      avgCosts,
      fixedMaxX,
      fixedMaxY,
      minX,
      stepBoundaries,
      visibleStepCosts,
    };
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

  // Place each closed-model label near the last visible point on that curve.
  const closedLabelPositions = useMemo(() => {
    const ANCHOR = 0.65;
    const minGap = fixedMaxY * 0.06;

    const entries = closedPricing.map((_, ci) => {
      const exitX = avgCosts[ci] > 0 ? fixedMaxY / avgCosts[ci] : fixedMaxX;
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
  }, [closedPricing, fixedMaxX, fixedMaxY, avgCosts]);

  const yAxisTicks = useMemo(
    () => niceYTicks(fixedMaxY, visibleStepCosts),
    [fixedMaxY, visibleStepCosts],
  );

  const sortedIntersections = useMemo(
    () => [...intersections].sort((a, b) => a.x - b.x),
    [intersections],
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
          Monthly cost at a given requests/month volume. Solid lines = API; dashed line = self-hosting
          (steps up when a new replica is needed). Dots mark the break-even point per pair.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-x-6 gap-y-3">
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
              tick={(props: { x: number; y: number; payload: { value: number } }) => {
                const { x, y, payload } = props;
                const isStepCost = visibleStepCosts.includes(payload.value);
                const label = `${currencySymbol}${
                  payload.value >= 1000
                    ? `${(payload.value / 1000).toFixed(0)}k`
                    : payload.value.toFixed(0)
                }`;
                return (
                  <text
                    x={x}
                    y={y}
                    dy={4}
                    textAnchor="end"
                    fontSize={11}
                    fontWeight={isStepCost ? 700 : 400}
                    fill={
                      isStepCost
                        ? (OPEN_MODEL_COLORS[0] ?? "#22c55e")
                        : "var(--muted-foreground)"
                    }
                  >
                    {label}
                  </text>
                );
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

            {openCosts[0]?.costConfig && (
              <Line
                type="linear"
                dataKey="selfHosting"
                stroke={OPEN_MODEL_COLORS[0] ?? "#22c55e"}
                strokeWidth={1.5}
                strokeDasharray="6 3"
                dot={false}
                name={openCosts[0].model.model_name}
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
                      {entry.model_name}
                    </text>
                  )}
                />
              );
            })}

            {stepBoundaries.map((boundary, i) => (
              <ReferenceLine
                key={`step-boundary-${boundary}`}
                x={boundary}
                stroke={OPEN_MODEL_COLORS[0] ?? "#22c55e"}
                strokeWidth={0.5}
                strokeOpacity={0.5}
                label={{
                  value: `×${i + 2}`,
                  position: "insideTop",
                  fontSize: 9,
                  fill: OPEN_MODEL_COLORS[0] ?? "#22c55e",
                  offset: 4,
                }}
              />
            ))}

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
                    {labToDisplayName(ix.closedModel.lab)} / {ix.openModel.model_name} (×{ix.replicaCount} replica):
                  </span>
                  <span className="font-medium tabular-nums">
                    {ix.x.toLocaleString("en-US", { maximumFractionDigits: 0 })} requests/mo
                  </span>
                  <span className="text-muted-foreground text-xs">
                    ({formatCost(ix.y)}/mo)
                  </span>
                  {ix.performanceNote && (
                    <span className="text-muted-foreground text-xs italic">
                      — {ix.performanceNote}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {openModels.length > 0 && openCosts.every(({ costConfig }) => costConfig == null) && (
          <p className="text-sm text-muted-foreground">
            No GPU offering found for the selected GPU configuration.
          </p>
        )}

        {availableModels.length > 0 && (
          <div className="text-sm space-y-1.5 pt-1 border-t">
            <p className="font-medium text-muted-foreground">Models served by this GPU config</p>
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              {availableModels.map((entry, i) => (
                <span key={entry.model.model_name} className="text-muted-foreground">
                  <span className={i === 0 ? "font-medium text-foreground" : ""}>
                    {entry.model.model_name}
                  </span>
                  {entry.sotaPercent !== null && (
                    <span className="text-xs ml-1">
                      ({Math.round(entry.sotaPercent)}% of SOTA)
                    </span>
                  )}
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
