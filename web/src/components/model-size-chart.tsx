"use client";

import { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  LabelList,
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
import type {
  ModelSizeScoreData,
  ModelSizeScorePoint,
  UnrankedModelSizePoint,
} from "@/lib/trend-data";
import { formatModelName } from "@/lib/utils";

const chartConfig = {
  score: {
    label: "Score",
    color: "var(--chart-2)",
  },
  unranked: {
    label: "Unranked",
    color: "var(--muted-foreground)",
  },
} satisfies ChartConfig;

interface ModelSizeChartProps {
  data: ModelSizeScoreData;
  categoryDisplayName: string;
}

function RankedTooltip({ active, payload }: {
  active?: boolean;
  payload?: Array<{
    payload: ModelSizeScorePoint;
  }>;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;

  return (
    <div className="rounded-lg border bg-background p-3 shadow-sm">
      <p className="text-sm font-medium mb-1.5">{formatModelName(point.modelName)}</p>
      <div className="space-y-1 text-sm">
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: "var(--chart-2)" }}
          />
          <span className="text-muted-foreground">Score:</span>
          <span className="font-medium">{point.score.toFixed(1)}</span>
        </div>
        <p className="text-muted-foreground text-xs">
          min {point.minVramGb} GB VRAM
        </p>
      </div>
    </div>
  );
}

function UnrankedTooltip({ active, payload }: {
  active?: boolean;
  payload?: Array<{
    payload: UnrankedModelSizePoint;
  }>;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;

  return (
    <div className="rounded-lg border bg-background p-3 shadow-sm">
      <p className="text-sm font-medium mb-1.5">{formatModelName(point.modelName)}</p>
      <div className="space-y-1 text-sm">
        <p className="text-muted-foreground">
          Unranked — no OpenHands Index result yet.
        </p>
        <p className="text-muted-foreground text-xs">
          min {point.minVramGb} GB VRAM
        </p>
      </div>
    </div>
  );
}

/**
 * Given an array of label positions ({x, y} in pixel space), compute
 * y-offsets so that overlapping labels are nudged apart vertically.
 * Labels that are within `minDx` horizontally and `minDy` vertically
 * of each other are considered overlapping.
 */
export function computeLabelOffsets(
  positions: Array<{ x: number; y: number; key: string }>,
  minDx = 40,
  minDy = 12,
): Map<string, number> {
  const offsets = new Map<string, number>();
  for (const p of positions) offsets.set(p.key, 0);

  // Sort by x so we compare nearby labels
  const sorted = [...positions].sort((a, b) => a.x - b.x || a.y - b.y);

  for (let i = 0; i < sorted.length; i++) {
    for (let j = i + 1; j < sorted.length; j++) {
      const dx = Math.abs(sorted[j].x - sorted[i].x);
      if (dx > minDx) break; // sorted by x, no further overlap possible

      const yI = sorted[i].y + (offsets.get(sorted[i].key) ?? 0);
      const yJ = sorted[j].y + (offsets.get(sorted[j].key) ?? 0);
      const dy = Math.abs(yJ - yI);

      if (dy < minDy) {
        // Nudge the second label down (away from the dot, further negative in SVG = up)
        offsets.set(sorted[j].key, (offsets.get(sorted[j].key) ?? 0) - (minDy - dy));
      }
    }
  }

  return offsets;
}

function makeModelLabel(fill: string, labelOffsets?: Map<string, number>) {
  function ModelLabel(props: Record<string, unknown>) {
    const { x, y, value } = props as { x: number; y: number; value: string };
    if (!value) return null;
    const offset = labelOffsets?.get(value) ?? 0;
    return (
      <text
        x={x}
        y={y - 10 + offset}
        textAnchor="middle"
        fontSize={10}
        fill={fill}
        className="select-none"
        pointerEvents="none"
      >
        {formatModelName(value)}
      </text>
    );
  }
  return ModelLabel;
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      <span className="text-muted-foreground">{label}</span>
    </div>
  );
}

export function ModelSizeChart({ data, categoryDisplayName }: ModelSizeChartProps) {
  const { ranked, unranked } = data;

  const yDomain = useMemo<[number, number]>(() => {
    if (ranked.length === 0) return [0, 100];
    const maxScore = Math.max(...ranked.map((p) => p.score));
    return [0, Math.ceil(maxScore * 1.15)];
  }, [ranked]);

  // X (VRAM) domain is shared by both series so the unranked strip lines up
  // with the ranked scatter above it.
  const xDomain = useMemo<[number, number]>(() => {
    const vrams = [
      ...ranked.map((p) => p.minVramGb),
      ...unranked.map((p) => p.minVramGb),
    ];
    if (vrams.length === 0) return [0, 100];
    return [0, Math.ceil(Math.max(...vrams) * 1.1)];
  }, [ranked, unranked]);

  // Approximate pixel positions from data coords and compute label offsets.
  // We use rough chart dimensions; the collision logic is tolerant of small errors.
  const rankedLabelOffsets = useMemo(() => {
    if (ranked.length === 0) return new Map<string, number>();
    const chartW = 500; // approximate plot area width
    const chartH = 300; // approximate plot area height
    const positions = ranked.map((p) => ({
      x: (p.minVramGb / xDomain[1]) * chartW,
      // SVG y is inverted: higher score → lower y
      y: (1 - p.score / yDomain[1]) * chartH,
      key: p.modelName,
    }));
    return computeLabelOffsets(positions);
  }, [ranked, xDomain, yDomain]);

  // The unranked strip has a single row, so labels only need horizontal
  // de-collision (all share the same y).
  const unrankedLabelOffsets = useMemo(() => {
    if (unranked.length === 0) return new Map<string, number>();
    const chartW = 500;
    const positions = unranked.map((p) => ({
      x: (p.minVramGb / xDomain[1]) * chartW,
      y: 0,
      key: p.modelName,
    }));
    return computeLabelOffsets(positions);
  }, [unranked, xDomain]);

  // The unranked series carries no score, so every point sits on a single
  // baseline; `y` is a constant placeholder, never a faked score.
  const unrankedBandData = useMemo(
    () => unranked.map((p) => ({ ...p, y: 1 })),
    [unranked],
  );

  if (ranked.length === 0 && unranked.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>How Big Are Open Models?</CardTitle>
          <CardDescription>
            Not enough data for this category.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>How Big Are Open Models?</CardTitle>
        <CardDescription>
          Minimum VRAM needed to serve each open-source model. Ranked models are
          placed by their score on {categoryDisplayName}; unranked models (size
          known, no OpenHands Index result yet) sit in the strip below, by size
          only.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="aspect-auto h-[300px] sm:h-[380px] w-full">
          <ScatterChart
            margin={{ top: 20, right: 30, bottom: 30, left: 10 }}
            accessibilityLayer
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis
              type="number"
              dataKey="minVramGb"
              name="Min VRAM"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              domain={xDomain}
              tickFormatter={(v: number) => `${v}`}
              label={{
                value: "Min VRAM (GB)",
                position: "insideBottom",
                offset: -15,
                style: { fill: "var(--muted-foreground)", fontSize: 12 },
              }}
            />
            <YAxis
              type="number"
              dataKey="score"
              name="Score"
              width={60}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              domain={yDomain}
              label={{
                value: categoryDisplayName,
                angle: -90,
                position: "insideLeft",
                style: {
                  fill: "var(--muted-foreground)",
                  fontSize: 12,
                  textAnchor: "middle",
                },
              }}
            />
            <ChartTooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={<RankedTooltip />}
            />
            <Scatter
              name="Ranked"
              data={ranked}
              fill="var(--chart-2)"
            >
              <LabelList
                dataKey="modelName"
                content={makeModelLabel(
                  "var(--chart-2)",
                  rankedLabelOffsets,
                ) as unknown as React.ReactElement}
              />
            </Scatter>
          </ScatterChart>
        </ChartContainer>

        {unranked.length > 0 && (
          <ChartContainer
            config={chartConfig}
            className="aspect-auto h-[90px] w-full"
          >
            <ScatterChart
              margin={{ top: 24, right: 30, bottom: 5, left: 10 }}
              accessibilityLayer
            >
              <XAxis
                type="number"
                dataKey="minVramGb"
                name="Min VRAM"
                domain={xDomain}
                hide
              />
              {/* Width matches the ranked chart's YAxis so the two share an X scale. */}
              <YAxis
                type="number"
                dataKey="y"
                width={60}
                domain={[0, 2]}
                tick={false}
                tickLine={false}
                axisLine={false}
              />
              <ChartTooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={<UnrankedTooltip />}
              />
              <Scatter
                name="Unranked"
                data={unrankedBandData}
                fill="var(--muted-foreground)"
                fillOpacity={0.6}
              >
                <LabelList
                  dataKey="modelName"
                  content={makeModelLabel(
                    "var(--muted-foreground)",
                    unrankedLabelOffsets,
                  ) as unknown as React.ReactElement}
                />
              </Scatter>
            </ScatterChart>
          </ChartContainer>
        )}

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
          <LegendItem color="var(--chart-2)" label="Ranked — size vs. score" />
          {unranked.length > 0 && (
            <LegendItem
              color="var(--muted-foreground)"
              label="Unranked — size known, score pending"
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
}
