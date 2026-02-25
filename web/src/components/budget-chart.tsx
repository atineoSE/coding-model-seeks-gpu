"use client";

import {
  ComposedChart,
  Bar,
  ErrorBar,
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
import type { BudgetChartDataPoint } from "@/lib/matrix-calculator";

const chartConfig = {
  teamSizeAvg: {
    label: "Avg team capacity",
    color: "var(--chart-1)",
  },
  percentOfSota: {
    label: "% of SOTA",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig;

/** Truncate model name for x-axis labels */
function truncateModel(name: string, maxLen: number = 18): string {
  if (name.length <= maxLen) return name;
  return name.slice(0, maxLen - 1) + "\u2026";
}

/** Compute evenly spaced "nice" ticks for a numeric range, targeting ~6-8 ticks. */
function niceTicksForRange(min: number, rawMax: number): number[] {
  if (rawMax <= 0) return [0];
  const rough = rawMax / 6;
  // Pick a nice step: 1, 2, 5, 10, 20, 50, ...
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const residual = rough / mag;
  let step: number;
  if (residual <= 1.5) step = mag;
  else if (residual <= 3.5) step = 2 * mag;
  else if (residual <= 7.5) step = 5 * mag;
  else step = 10 * mag;
  // Round step to avoid floating-point dust for small ranges
  if (step < 1) step = Math.round(step * 100) / 100;
  const ticks: number[] = [];
  for (let v = min; v <= rawMax + step * 0.01; v += step) {
    ticks.push(Math.round(v * 100) / 100);
  }
  // Ensure at least one tick above rawMax for headroom
  if (ticks[ticks.length - 1] < rawMax) {
    ticks.push(Math.round((ticks[ticks.length - 1] + step) * 100) / 100);
  }
  return ticks;
}

interface BudgetChartProps {
  data: BudgetChartDataPoint[];
}

export function BudgetChart({ data }: BudgetChartProps) {
  const nonFittingModels = data.filter((d) => !d.fits);

  // Prepare chart data — all models, non-fitting ones show empty bars
  const chartData = data.map((d) => ({
    ...d,
    modelLabel: truncateModel(d.modelName),
    // Asymmetric error: [below average (to CLI), above average (to IDE)]; skip for non-fitting models
    teamSizeError: d.fits
      ? ([d.teamSizeAvg - d.teamSizeCli, d.teamSizeIde - d.teamSizeAvg] as [number, number])
      : undefined,
  }));

  if (chartData.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No models available for the selected benchmark category.
      </div>
    );
  }

  // Dynamic left Y axis domain with proportional ticks
  const maxTeamSize = Math.max(...chartData.map((d) => d.teamSizeIde), 1);
  const teamTicks = niceTicksForRange(0, maxTeamSize * 1.1);
  const yTeamMax = teamTicks[teamTicks.length - 1];

  return (
    <div>
      {chartData.length > 0 && (
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
              yAxisId="team"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              domain={[0, yTeamMax]}
              ticks={teamTicks}
              label={{
                value: "Development Team Size",
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
            />
            <ChartTooltip content={<BudgetTooltip />} />
            <Legend content={<BudgetLegend />} />

            {/* Single bar: average team capacity with CI error bars */}
            <Bar dataKey="teamSizeAvg" yAxisId="team" name="teamSizeAvg" barSize={28}>
              {chartData.map((_, i) => (
                <Cell key={i} fill="var(--chart-1)" />
              ))}
              <ErrorBar dataKey="teamSizeError" width={8} strokeWidth={1.5} stroke="var(--foreground)" />
            </Bar>

            {/* Line: % of SOTA performance */}
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
      )}
      {nonFittingModels.length > 0 && (
        <p className="text-xs text-muted-foreground mt-2 px-1">
          The current GPU setup cannot accommodate{" "}
          {formatModelList(nonFittingModels.map((m) => m.modelName))}. Try a larger setup.
        </p>
      )}
    </div>
  );
}

/** Format a list of model names with commas and "and" */
function formatModelList(names: string[]): string {
  if (names.length === 1) return names[0];
  if (names.length === 2) return `${names[0]} and ${names[1]}`;
  return `${names.slice(0, -1).join(", ")}, and ${names[names.length - 1]}`;
}

/** Custom tooltip */
function BudgetTooltip({ active, payload }: {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    payload: BudgetChartDataPoint & { teamSizeRange: number };
  }>;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;

  return (
    <div className="rounded-lg border bg-background p-3 shadow-sm">
      <div className="space-y-1.5 text-sm">
        <p className="font-medium">{point.modelName}</p>
        {point.fits ? (
          <>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-muted-foreground">
              <span>Team size (IDE):</span>
              <span className="font-medium text-foreground">{point.teamSizeIde} devs</span>
              <span>Team size (CLI):</span>
              <span className="font-medium text-foreground">{point.teamSizeCli} devs</span>
              <span>Team size (avg):</span>
              <span className="font-medium text-foreground">{point.teamSizeAvg} devs</span>
            </div>
            <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-muted-foreground pt-1 border-t">
              <span>% of SOTA:</span>
              <span className="font-medium text-foreground">{point.percentOfSota.toFixed(1)}%</span>
              <span>Streams:</span>
              <span className="font-medium text-foreground">{point.concurrentStreams} / {point.maxConcurrentStreams}</span>
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
          <p className="text-muted-foreground">
            {point.doesntFitReason ?? "Model doesn't fit"}
          </p>
        )}
      </div>
    </div>
  );
}

/** Custom legend */
function BudgetLegend() {
  return (
    <div className="flex flex-wrap items-center justify-center gap-4 pt-3 text-xs">
      {/* Avg capacity bar */}
      <div className="flex items-center gap-1.5">
        <div className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: "var(--chart-1)" }} />
        <span className="text-muted-foreground">Avg team capacity</span>
      </div>
      {/* CI whisker */}
      <div className="flex items-center gap-1.5">
        <svg width="16" height="12" viewBox="0 0 16 12" className="shrink-0">
          <line x1="8" y1="1" x2="8" y2="11" stroke="var(--foreground)" strokeWidth="1.5" />
          <line x1="4" y1="1" x2="12" y2="1" stroke="var(--foreground)" strokeWidth="1.5" />
          <line x1="4" y1="11" x2="12" y2="11" stroke="var(--foreground)" strokeWidth="1.5" />
        </svg>
        <span className="text-muted-foreground">CLI / IDE range</span>
      </div>
      {/* SOTA line */}
      <div className="flex items-center gap-1.5">
        <div className="h-0.5 w-4 rounded" style={{ backgroundColor: "var(--chart-2)" }} />
        <span className="text-muted-foreground">% of SOTA</span>
      </div>
    </div>
  );
}
