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
import type { ModelSizeScorePoint } from "@/lib/trend-data";

const chartConfig = {
  score: {
    label: "Score",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

interface ModelSizeChartProps {
  data: ModelSizeScorePoint[];
  categoryDisplayName: string;
}

function CustomTooltip({ active, payload }: {
  active?: boolean;
  payload?: Array<{
    payload: ModelSizeScorePoint;
  }>;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;

  return (
    <div className="rounded-lg border bg-background p-3 shadow-sm">
      <p className="text-sm font-medium mb-1.5">{point.modelName}</p>
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

function ModelLabel(props: Record<string, unknown>) {
  const { x, y, value } = props as { x: number; y: number; value: string };
  if (!value) return null;
  return (
    <text
      x={x}
      y={y - 10}
      textAnchor="middle"
      fontSize={10}
      fill="var(--chart-2)"
      className="select-none"
    >
      {value}
    </text>
  );
}

export function ModelSizeChart({ data, categoryDisplayName }: ModelSizeChartProps) {
  const yDomain = useMemo<[number, number]>(() => {
    if (data.length === 0) return [0, 100];
    const maxScore = Math.max(...data.map((p) => p.score));
    return [0, Math.ceil(maxScore * 1.15)];
  }, [data]);

  const xDomain = useMemo<[number, number]>(() => {
    if (data.length === 0) return [0, 100];
    const maxVram = Math.max(...data.map((p) => p.minVramGb));
    return [0, Math.ceil(maxVram * 1.1)];
  }, [data]);

  if (data.length === 0) {
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
          Minimum VRAM needed to serve each open-source model vs. its score on {categoryDisplayName}.
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
              content={<CustomTooltip />}
            />
            <Scatter
              name="Open source models"
              data={data}
              fill="var(--chart-2)"
            >
              <LabelList dataKey="modelName" content={<ModelLabel />} />
            </Scatter>
          </ScatterChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
