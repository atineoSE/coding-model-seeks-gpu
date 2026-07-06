"use client";

import { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
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
import type { GpuNodePriceTrendRow } from "@/lib/gpu-trend-data";

// Every node is the 8-GPU serving tier; the chart tracks the whole-node
// (bundle) price, so labels read e.g. "8× H100".
const NODE_GPU_COUNT = 8;

// One <Line> per GPU node, in draw order. Colors cycle through the five chart
// palette slots (there are more nodes than palette colors).
const NODES = [
  { key: "B300", color: "var(--chart-1)" },
  { key: "B200", color: "var(--chart-2)" },
  { key: "H200", color: "var(--chart-3)" },
  { key: "H100", color: "var(--chart-4)" },
  { key: "A100", color: "var(--chart-5)" },
  { key: "RTXPRO6000", color: "var(--chart-1)" },
] as const;

const chartConfig = Object.fromEntries(
  NODES.map((n) => [n.key, { label: n.key, color: n.color }]),
) satisfies ChartConfig;

interface GpuNodePriceChartProps {
  data: GpuNodePriceTrendRow[];
  currencySymbol?: string;
}

interface GpuNodePriceTooltipProps {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    color: string;
    payload: GpuNodePriceTrendRow;
  }>;
  label?: string;
  currencySymbol: string;
}

function CustomTooltip({
  active,
  payload,
  label,
  currencySymbol,
}: GpuNodePriceTooltipProps) {
  if (!active || !payload?.length) return null;

  // Sort most-expensive node first so the overlay reads high → low.
  const rows = [...payload].sort((a, b) => b.value - a.value);

  return (
    <div className="rounded-lg border bg-background p-3 shadow-sm">
      <p className="text-sm font-medium mb-1.5">{label}</p>
      <div className="space-y-1 text-sm">
        {rows.map((entry) => (
          <div key={entry.dataKey} className="flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-muted-foreground">
              {NODE_GPU_COUNT}× {entry.dataKey}:
            </span>
            <span className="font-medium">
              {currencySymbol}
              {entry.value.toLocaleString("en-US", {
                maximumFractionDigits: 0,
              })}
              /mo
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function GpuNodePriceChart({
  data,
  currencySymbol = "$",
}: GpuNodePriceChartProps) {
  const dates = useMemo(() => data.map((row) => row.date), [data]);

  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Are GPU Nodes Getting Cheaper to Rent?</CardTitle>
          <CardDescription>Not enough data yet.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Are GPU Nodes Getting Cheaper to Rent?</CardTitle>
        <CardDescription>
          Cheapest monthly rental cost for an 8× GPU node, tracked at each price
          change.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-[300px] sm:h-[380px] w-full"
        >
          <LineChart
            data={data}
            margin={{ top: 5, right: 60, bottom: 5, left: 5 }}
            accessibilityLayer
          >
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              ticks={dates}
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
              tickFormatter={(v: number) =>
                `${currencySymbol}${(v / 1000).toFixed(v >= 1000 ? 0 : 1)}k/mo`
              }
            />
            <ChartTooltip
              content={<CustomTooltip currencySymbol={currencySymbol} />}
            />
            {NODES.map((node) => (
              <Line
                key={node.key}
                type="monotone"
                dataKey={node.key}
                stroke={node.color}
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls={false}
                name={node.key}
              />
            ))}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
