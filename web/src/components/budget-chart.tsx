"use client";

import { useState, useMemo } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
  Legend,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  type ChartConfig,
} from "@/components/ui/chart";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import type { BudgetChartDataPoint } from "@/lib/matrix-calculator";
import { formatModelName } from "@/lib/utils";

const DEFAULT_REQ_PER_DEV_HOUR = 200;
const ZOOMED_MODEL_COUNT = 6;

const chartConfig = {
  displayValue: {
    label: "Capacity",
    color: "var(--chart-1)",
  },
  percentOfSota: {
    label: "% of SOTA",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

function truncateModel(name: string, maxLen: number = 18): string {
  if (name.length <= maxLen) return name;
  return name.slice(0, maxLen - 1) + "…";
}

function formatReqH(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(v % 1000 === 0 ? 0 : 1)}k`;
  return String(Math.round(v));
}

function niceTicksForRange(min: number, rawMax: number): number[] {
  if (rawMax <= 0) return [0];
  const rough = rawMax / 6;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const residual = rough / mag;
  let step: number;
  if (residual <= 1.5) step = mag;
  else if (residual <= 3.5) step = 2 * mag;
  else if (residual <= 7.5) step = 5 * mag;
  else step = 10 * mag;
  if (step < 1) step = Math.round(step * 100) / 100;
  const ticks: number[] = [];
  for (let v = min; v <= rawMax + step * 0.01; v += step) {
    ticks.push(Math.round(v * 100) / 100);
  }
  if (ticks[ticks.length - 1] < rawMax) {
    ticks.push(Math.round((ticks[ticks.length - 1] + step) * 100) / 100);
  }
  return ticks;
}

interface BudgetChartProps {
  data: BudgetChartDataPoint[];
}

export function BudgetChart({ data }: BudgetChartProps) {
  const [mode, setMode] = useState<"request" | "team">("request");
  const [reqPerDevPerHour, setReqPerDevPerHour] = useState(DEFAULT_REQ_PER_DEV_HOUR);
  const [zoomed, setZoomed] = useState(false);

  const visibleData = zoomed ? data : data.slice(0, ZOOMED_MODEL_COUNT);
  const nonFittingModels = visibleData.filter((d) => !d.fits);

  const chartData = useMemo(() =>
    visibleData.map((d) => {
      let displayValue = 0;
      if (d.fits && d.requestsPerHour !== null) {
        displayValue = mode === "request"
          ? d.requestsPerHour
          : Math.floor(d.requestsPerHour / reqPerDevPerHour);
      }
      return { ...d, modelLabel: truncateModel(formatModelName(d.modelName)), displayValue };
    }),
    [visibleData, mode, reqPerDevPerHour],
  );

  if (chartData.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No models available for the selected benchmark category.
      </div>
    );
  }

  const maxDisplayValue = Math.max(...chartData.map((d) => d.displayValue), 1);
  const yTicks = niceTicksForRange(0, maxDisplayValue * 1.1);
  const yMax = yTicks[yTicks.length - 1];
  const yLabel = mode === "request" ? "Requests / h" : "Team size (devs)";

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex rounded-md border overflow-hidden text-sm">
          <button
            onClick={() => setMode("request")}
            className={`px-3 py-1.5 transition-colors ${mode === "request" ? "bg-primary text-primary-foreground" : "hover:bg-accent cursor-pointer"}`}
          >
            Request capacity
          </button>
          <button
            onClick={() => setMode("team")}
            className={`px-3 py-1.5 transition-colors border-l ${mode === "team" ? "bg-primary text-primary-foreground" : "hover:bg-accent cursor-pointer"}`}
          >
            Team capacity
          </button>
        </div>

        {mode === "team" && (
          <div className="flex items-center gap-3">
            <Label className="text-sm whitespace-nowrap text-muted-foreground">
              {reqPerDevPerHour} req/dev/h
            </Label>
            <Slider
              value={[reqPerDevPerHour]}
              onValueChange={([v]) => setReqPerDevPerHour(v)}
              min={100}
              max={500}
              step={10}
              className="w-36"
            />
          </div>
        )}

        {data.length > ZOOMED_MODEL_COUNT && (
          <button
            onClick={() => setZoomed((z) => !z)}
            className="ml-auto flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            title={zoomed ? "Zoom in" : "Zoom out"}
          >
            {zoomed ? (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="6" cy="6" r="4" />
                <path d="M9 9l3 3M4 6h4M6 4v4" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="6" cy="6" r="4" />
                <path d="M9 9l3 3M4 6h4" />
              </svg>
            )}
            {zoomed ? "Zoom in" : "Zoom out"}
          </button>
        )}
      </div>

      <ChartContainer config={chartConfig} className="aspect-auto h-[350px] sm:h-[420px] w-full">
        <ComposedChart data={chartData} accessibilityLayer margin={{ top: 24, right: 12, bottom: 60, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" vertical={false} />
          <XAxis
            dataKey="modelLabel"
            tickLine={false}
            axisLine={false}
            interval={0}
            angle={-45}
            textAnchor="end"
            height={80}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            yAxisId="left"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            domain={[0, yMax]}
            ticks={yTicks}
            tickFormatter={formatReqH}
            label={{
              value: yLabel,
              angle: -90,
              position: "insideLeft",
              offset: -8,
              fontSize: 12,
              fill: "var(--muted-foreground)",
            }}
          />
          <YAxis
            yAxisId="sota"
            orientation="right"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            domain={[0, 100]}
            tickFormatter={(v: number) => `${v}%`}
            label={{
              value: "Benchmark score",
              angle: 90,
              position: "insideRight",
              offset: 12,
              fontSize: 12,
              fill: "var(--muted-foreground)",
            }}
          />
          <ChartTooltip content={<BudgetTooltip mode={mode} reqPerDevPerHour={reqPerDevPerHour} />} />
          <Legend content={<BudgetLegend mode={mode} />} />

          <Bar dataKey="displayValue" yAxisId="left" name="displayValue" barSize={28}>
            {chartData.map((_, i) => (
              <Cell key={i} fill="var(--chart-1)" />
            ))}
          </Bar>

          <Line
            yAxisId="sota"
            type="monotone"
            dataKey="percentOfSota"
            stroke="var(--chart-2)"
            strokeWidth={2}
            dot={{ r: 3 }}
            name="percentOfSota"
          />
        </ComposedChart>
      </ChartContainer>

      {nonFittingModels.length > 0 && (
        <p className="text-xs text-muted-foreground px-1">
          The current GPU setup cannot accommodate{" "}
          {formatModelList(nonFittingModels.map((m) => formatModelName(m.modelName)))}. Try a larger setup.
        </p>
      )}
    </div>
  );
}

function formatModelList(names: string[]): string {
  if (names.length === 1) return names[0];
  if (names.length === 2) return `${names[0]} and ${names[1]}`;
  return `${names.slice(0, -1).join(", ")}, and ${names[names.length - 1]}`;
}

function BudgetTooltip({ active, payload, mode, reqPerDevPerHour }: {
  active?: boolean;
  payload?: Array<{ dataKey: string; value: number; payload: BudgetChartDataPoint & { displayValue: number } }>;
  mode: "request" | "team";
  reqPerDevPerHour: number;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;

  return (
    <div className="rounded-lg border bg-background p-3 shadow-sm">
      <div className="space-y-1.5 text-sm">
        <p className="font-medium">{formatModelName(point.modelName)}</p>
        {point.fits ? (
          <>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-muted-foreground">
              {mode === "request" ? (
                <>
                  <span>Requests / h:</span>
                  <span className="font-medium text-foreground">
                    {point.requestsPerHour !== null ? formatReqH(point.requestsPerHour) : "—"}
                  </span>
                </>
              ) : (
                <>
                  <span>Team size:</span>
                  <span className="font-medium text-foreground">{point.displayValue} devs</span>
                  <span>at {reqPerDevPerHour} req/dev/h</span>
                  <span className="font-medium text-foreground">
                    {point.requestsPerHour !== null ? formatReqH(point.requestsPerHour) : "—"} req/h capacity
                  </span>
                </>
              )}
            </div>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-muted-foreground pt-1 border-t">
              <span>% of SOTA:</span>
              <span className="font-medium text-foreground">{point.percentOfSota.toFixed(1)}%</span>
              <span>Streams:</span>
              <span className="font-medium text-foreground">{point.maxConcurrentStreams}</span>
              <span>Memory:</span>
              <span className="font-medium text-foreground">{point.modelMemoryGb.toFixed(0)} GB</span>
              {point.decodeThroughputTokS !== null && (
                <>
                  <span>Decode:</span>
                  <span className="font-medium text-foreground">{Math.round(point.decodeThroughputTokS)} tok/s</span>
                </>
              )}
            </div>
          </>
        ) : (
          <p className="text-muted-foreground">{point.doesntFitReason ?? "Model doesn't fit"}</p>
        )}
      </div>
    </div>
  );
}

function BudgetLegend({ mode }: { mode: "request" | "team" }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-4 pt-3 text-xs">
      <div className="flex items-center gap-1.5">
        <div className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "var(--chart-1)" }} />
        <span className="text-muted-foreground">{mode === "request" ? "Requests / h" : "Team size"}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="h-0.5 w-4 rounded" style={{ backgroundColor: "var(--chart-2)" }} />
        <span className="text-muted-foreground">% of SOTA</span>
      </div>
    </div>
  );
}

