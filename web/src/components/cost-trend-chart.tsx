"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
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
import {
  type CostTrendPoint,
  type GpuReferenceCost,
} from "@/lib/trend-data";

const chartConfig = {
  monthlyCost: {
    label: "Monthly Cost",
    color: "var(--chart-3)",
  },
} satisfies ChartConfig;

interface CostTrendChartProps {
  data: CostTrendPoint[];
  referenceCosts: GpuReferenceCost[];
  currencySymbol?: string;
}

/** Chart point with a pre-computed display label (blank on carry-forward). */
interface ChartPoint extends CostTrendPoint {
  displayLabel: string;
  isFirst: boolean;
  isLast: boolean;
}

function CostLabel(props: Record<string, unknown>) {
  const { x, y, value } = props as { x: number; y: number; value: string };
  const point = (props as { payload?: ChartPoint }).payload;
  if (!value) return null;
  const anchor = point?.isFirst ? "start" : point?.isLast ? "end" : "middle";
  return (
    <text
      x={x}
      y={y - 14}
      textAnchor={anchor}
      fontSize={9}
      fill="var(--chart-3)"
      className="select-none"
    >
      {value}
    </text>
  );
}

const REFERENCE_COLORS = [
  "var(--muted-foreground)",
  "var(--muted-foreground)",
  "var(--muted-foreground)",
  "var(--muted-foreground)",
  "var(--muted-foreground)",
];

export function CostTrendChart({ data, referenceCosts, currencySymbol = "$" }: CostTrendChartProps) {
  function CustomTooltip({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{
      value: number;
      payload: CostTrendPoint;
    }>;
    label?: string;
  }) {
    if (!active || !payload?.length) return null;
    const point = payload[0].payload;

    return (
      <div className="rounded-lg border bg-background p-3 shadow-sm">
        <p className="text-sm font-medium mb-1.5">{label}</p>
        <div className="space-y-1 text-sm">
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: "var(--chart-3)" }}
            />
            <span className="font-medium">
              {currencySymbol}{point.monthlyCost.toLocaleString("en-US", { maximumFractionDigits: 0 })}/mo
            </span>
          </div>
          <p className="text-muted-foreground">
            {point.modelName} &middot; {point.gpuSetup}
          </p>
          <p className="text-muted-foreground text-xs">
            Score: {point.score.toFixed(1)} &middot; Weights: {point.modelMemoryGb.toFixed(0)} GB
          </p>
        </div>
      </div>
    );
  }

  const chartData = useMemo(() => {
    return data.map((p, i): ChartPoint => ({
      ...p,
      displayLabel: `${p.modelName} (${p.score.toFixed(1)}, ${p.modelMemoryGb.toFixed(0)} gb)`,
      isFirst: i === 0,
      isLast: i === data.length - 1,
    }));
  }, [data]);

  const changePointDates = useMemo(
    () => chartData.map((p) => p.date),
    [chartData],
  );

  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Is It Getting Cheaper to Run the Best?</CardTitle>
          <CardDescription>
            Not enough data for this category.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  // Compute Y domain: include both data and reference lines
  const allCosts = [
    ...chartData.map((d) => d.monthlyCost),
    ...referenceCosts.map((r) => r.monthlyCost),
  ];
  const maxCost = Math.max(...allCosts);
  const minCost = Math.min(...allCosts);
  const yPadding = (maxCost - minCost) * 0.1 || 500;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Is It Getting Cheaper to Run the Best?</CardTitle>
        <CardDescription>
          Monthly cloud cost for the top open source coding LLM over time (5 concurrent streams).
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[380px] w-full">
          <LineChart
            data={chartData}
            margin={{ top: 20, right: 60, bottom: 5, left: 5 }}
            accessibilityLayer
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              ticks={changePointDates}
              interval={0}
              angle={-30}
              textAnchor="end"
              height={50}
              tickFormatter={(v: string) => {
                const d = new Date(v + "T00:00:00");
                return d.toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                });
              }}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              domain={[
                Math.max(0, minCost - yPadding),
                maxCost + yPadding,
              ]}
              tickFormatter={(v: number) =>
                `${currencySymbol}${(v / 1000).toFixed(v >= 1000 ? 0 : 1)}k`
              }
            />
            <ChartTooltip content={<CustomTooltip />} />
            {referenceCosts.map((ref, i) => (
              <ReferenceLine
                key={ref.label}
                y={ref.monthlyCost}
                stroke={REFERENCE_COLORS[i % REFERENCE_COLORS.length]}
                strokeDasharray="6 3"
                strokeOpacity={0.5}
                label={{
                  value: ref.label,
                  position: "right",
                  fontSize: 11,
                  fill: "var(--muted-foreground)",
                }}
              />
            ))}
            <Line
              type="monotone"
              dataKey="monthlyCost"
              stroke="var(--chart-3)"
              strokeWidth={2}
              dot={{ r: 4 }}
              name="Monthly Cost"
            >
              <LabelList
                dataKey="displayLabel"
                content={<CostLabel />}
              />
            </Line>
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
