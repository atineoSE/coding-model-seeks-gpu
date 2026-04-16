"use client";

import { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, LabelList } from "recharts";
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
import type { SotaPercentTrendPoint } from "@/lib/trend-data";

const chartConfig = {
  percentOfSota: {
    label: "Percent of SOTA",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

interface SotaPercentChartProps {
  data: SotaPercentTrendPoint[];
}

/** Chart point with pre-scaled percent value and positional flags. */
interface ChartPoint extends SotaPercentTrendPoint {
  percentDisplay: number; // 0–100
  modelLabel: string;
  isFirst: boolean;
  isLast: boolean;
}

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{
    value: number;
    payload: ChartPoint;
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
            style={{ backgroundColor: "var(--chart-2)" }}
          />
          <span className="font-medium">
            {point.percentDisplay.toFixed(1)}%
          </span>
        </div>
        <p className="text-muted-foreground">{point.openSourceModel}</p>
      </div>
    </div>
  );
}

function ModelLabel(props: Record<string, unknown>) {
  const { x, y, value } = props as { x: number; y: number; value: string };
  const point = (props as { payload?: ChartPoint }).payload;
  if (!value) return null;
  const anchor = point?.isFirst ? "start" : point?.isLast ? "end" : "middle";
  return (
    <text
      x={x}
      y={y - 14}
      textAnchor={anchor}
      fontSize={10}
      fill="var(--chart-2)"
      className="select-none"
    >
      {value}
    </text>
  );
}

export function SotaPercentChart({ data }: SotaPercentChartProps) {
  const chartData = useMemo(() => {
    return data.map((p, i): ChartPoint => ({
      ...p,
      percentDisplay: p.percentOfSota * 100,
      modelLabel: p.openSourceModel,
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
          <CardTitle>Percentage of SOTA for the Best Open Model</CardTitle>
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
        <CardTitle>Percentage of SOTA for the Best Open Model</CardTitle>
        <CardDescription>
          Best open source model score as a fraction of the best closed source model, over time.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="aspect-auto h-[300px] sm:h-[380px] w-full">
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
              domain={[0, 100]}
              tickFormatter={(v: number) => `${v}%`}
            />
            <ChartTooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="percentDisplay"
              stroke="var(--chart-2)"
              strokeWidth={2}
              dot={{ r: 4 }}
              name="Percent of SOTA"
            >
              <LabelList
                dataKey="modelLabel"
                content={<ModelLabel />}
              />
            </Line>
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
