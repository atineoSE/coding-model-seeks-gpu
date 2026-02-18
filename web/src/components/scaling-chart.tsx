"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
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
import type { ScalingCurvePoint, GpuReferenceCost } from "@/lib/trend-data";

const chartConfig = {
  monthlyCost: {
    label: "Monthly Cost",
    color: "var(--chart-4)",
  },
  gpuUtilisation: {
    label: "GPU Utilisation",
    color: "var(--chart-5)",
  },
} satisfies ChartConfig;

interface ScalingChartProps {
  data: ScalingCurvePoint[];
  referenceCosts: GpuReferenceCost[];
  modelName: string;
  categoryDisplayName: string;
  currencySymbol?: string;
}

export function ScalingChart({ data, referenceCosts, modelName, categoryDisplayName, currencySymbol = "$" }: ScalingChartProps) {
  function CustomTooltip({ active, payload, label }: {
    active?: boolean;
    payload?: Array<{
      dataKey: string;
      value: number;
      payload: ScalingCurvePoint;
    }>;
    label?: string;
  }) {
    if (!active || !payload?.length) return null;
    const point = payload[0].payload;

    return (
      <div className="rounded-lg border bg-background p-3 shadow-sm">
        <div className="space-y-1 text-sm">
          <p className="font-medium">
            {label} concurrent stream{Number(label) !== 1 ? "s" : ""}
          </p>
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: "var(--chart-4)" }}
            />
            <span className="font-medium">
              {currencySymbol}{point.monthlyCost.toLocaleString("en-US", { maximumFractionDigits: 0 })}/mo
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: "var(--chart-5)" }}
            />
            <span className="font-medium">
              {point.gpuUtilisation}% utilisation
            </span>
          </div>
          <p className="text-muted-foreground">{point.gpuSetup}</p>
        </div>
      </div>
    );
  }
  if (data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>What Does It Cost to Scale Agents?</CardTitle>
          <CardDescription>
            No scaling data available.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const dataCosts = data.map((d) => d.monthlyCost);
  const maxDataCost = Math.max(...dataCosts);
  const minDataCost = Math.min(...dataCosts);
  const yPadding = (maxDataCost - minDataCost) * 0.15 || 500;

  // Only show reference lines that fall within the visible data range
  const yMin = Math.max(0, minDataCost - yPadding);
  const yMax = maxDataCost + yPadding;
  const visibleReferenceCosts = referenceCosts.filter(
    (r) => r.monthlyCost >= yMin && r.monthlyCost <= yMax,
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>What Does It Cost to Scale Agents?</CardTitle>
        <CardDescription>
          Monthly cloud cost as you add concurrent coding streams. Showing the best open source {categoryDisplayName} model, {modelName}.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[300px] w-full">
          <LineChart data={data} accessibilityLayer>
            <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
            <XAxis
              dataKey="concurrency"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              label={{
                value: "Concurrent Streams",
                position: "insideBottom",
                offset: -2,
                fontSize: 12,
                fill: "var(--muted-foreground)",
              }}
            />
            <YAxis
              yAxisId="cost"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              domain={[yMin, yMax]}
              tickFormatter={(v: number) =>
                `${currencySymbol}${(v / 1000).toFixed(v >= 1000 ? 0 : 1)}k`
              }
            />
            <YAxis
              yAxisId="util"
              orientation="right"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              domain={[0, 100]}
              tickFormatter={(v: number) => `${v}%`}
            />
            <ChartTooltip content={<CustomTooltip />} />
            {visibleReferenceCosts.map((ref) => (
              <ReferenceLine
                key={ref.label}
                yAxisId="cost"
                y={ref.monthlyCost}
                stroke="var(--muted-foreground)"
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
              yAxisId="cost"
              type="monotone"
              dataKey="monthlyCost"
              stroke="var(--chart-4)"
              strokeWidth={2}
              dot={{ r: 3 }}
              name="Monthly Cost"
            />
            <Line
              yAxisId="util"
              type="monotone"
              dataKey="gpuUtilisation"
              stroke="var(--chart-5)"
              strokeWidth={2}
              strokeDasharray="4 2"
              dot={{ r: 2 }}
              name="GPU Utilisation"
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
