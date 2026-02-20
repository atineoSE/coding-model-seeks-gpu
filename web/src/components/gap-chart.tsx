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
import type { GapTrendPoint } from "@/lib/trend-data";

const chartConfig = {
  closedSourceScore: {
    label: "Closed Source",
    color: "var(--chart-1)",
  },
  openSourceScore: {
    label: "Open Source",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

interface GapChartProps {
  data: GapTrendPoint[];
}

/** Deduplicated point with label fields that are blank on carry-forward points. */
interface ChartPoint extends GapTrendPoint {
  closedLabel: string;
  openLabel: string;
  isFirst: boolean;
  isLast: boolean;
}

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    payload: GapTrendPoint;
    color: string;
  }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-lg border bg-background p-3 shadow-sm">
      <p className="text-sm font-medium mb-1.5">{label}</p>
      {payload.map((entry) => {
        const point = entry.payload;
        const modelName =
          entry.dataKey === "closedSourceScore"
            ? point.closedSourceModel
            : point.openSourceModel;
        const isOpen = entry.dataKey === "openSourceScore";
        return (
          <div key={entry.dataKey} className="flex items-center gap-2 text-sm">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-muted-foreground">
              {isOpen ? "Open" : "Closed"}:
            </span>
            <span className="font-medium">{entry.value.toFixed(1)}</span>
            <span className="text-muted-foreground text-xs">
              ({modelName})
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ClosedSourceLabel(props: Record<string, unknown>) {
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
      fill="var(--chart-1)"
      className="select-none"
    >
      {value}
    </text>
  );
}

function OpenSourceLabel(props: Record<string, unknown>) {
  const { x, y, value } = props as { x: number; y: number; value: string };
  const point = (props as { payload?: ChartPoint }).payload;
  if (!value) return null;
  const anchor = point?.isFirst ? "start" : point?.isLast ? "end" : "middle";
  return (
    <text
      x={x}
      y={y + 20}
      textAnchor={anchor}
      fontSize={10}
      fill="var(--chart-2)"
      className="select-none"
    >
      {value}
    </text>
  );
}

export function GapChart({ data }: GapChartProps) {
  const chartData = useMemo(() => {
    return data.map((p, i): ChartPoint => ({
      ...p,
      closedLabel: p.closedSourceModel,
      openLabel: p.openSourceModel,
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
          <CardTitle>Is Open Source Closing the Gap?</CardTitle>
          <CardDescription>
            Not enough data for this category.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const yDomain: [number, number] = [0, 100];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Is Open Source Closing the Gap?</CardTitle>
        <CardDescription>
          Best closed vs. open source model scores over time.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="aspect-auto h-[300px] sm:h-[380px] w-full">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 60, bottom: 5, left: 5 }}
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
              domain={yDomain}
            />
            <ChartTooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="closedSourceScore"
              stroke="var(--chart-1)"
              strokeWidth={2}
              dot={{ r: 4 }}
              name="Closed Source"
            >
              <LabelList
                dataKey="closedLabel"
                content={<ClosedSourceLabel />}
              />
            </Line>
            <Line
              type="monotone"
              dataKey="openSourceScore"
              stroke="var(--chart-2)"
              strokeWidth={2}
              dot={{ r: 4 }}
              name="Open Source"
            >
              <LabelList
                dataKey="openLabel"
                content={<OpenSourceLabel />}
              />
            </Line>
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
