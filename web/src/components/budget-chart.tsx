"use client";

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
import type { BudgetChartDataPoint } from "@/lib/matrix-calculator";

const chartConfig = {
  teamSizeCli: {
    label: "CLI-workflow capacity",
    color: "var(--chart-1)",
  },
  teamSizeRange: {
    label: "IDE-workflow capacity",
    color: "var(--chart-1)",
  },
  doesntFit: {
    label: "Model doesn't fit",
    color: "var(--muted)",
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

interface BudgetChartProps {
  data: BudgetChartDataPoint[];
}

export function BudgetChart({ data }: BudgetChartProps) {
  // Prepare chart data with the stacked range field
  const chartData = data.map((d) => ({
    ...d,
    modelLabel: truncateModel(d.modelName),
    // Bottom segment: CLI capacity (lower bound)
    teamSizeCli: d.teamSizeCli,
    // Top segment: extension from CLI to IDE capacity
    teamSizeRange: Math.max(0, d.teamSizeIde - d.teamSizeCli),
  }));

  if (chartData.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No models available for the selected benchmark category.
      </div>
    );
  }

  // Dynamic left Y axis domain based on max team size
  const maxTeamSize = Math.max(...chartData.map((d) => d.teamSizeIde), 1);
  const yTeamMax = Math.ceil(maxTeamSize * 1.1);

  return (
    <ChartContainer config={chartConfig} className="aspect-auto h-[350px] sm:h-[420px] w-full">
      <ComposedChart data={chartData} accessibilityLayer margin={{ top: 8, right: 12, bottom: 60, left: 4 }}>
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
          label={{
            value: "Development Team Size",
            angle: -90,
            position: "insideLeft",
            offset: 10,
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

        {/* Bottom segment: CLI capacity (the baseline) */}
        <Bar dataKey="teamSizeCli" yAxisId="team" stackId="team" name="teamSizeCli" barSize={28}>
          {chartData.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.fits ? "var(--chart-1)" : "var(--muted)"}
            />
          ))}
        </Bar>
        {/* Top segment: extension from CLI to IDE (the range) */}
        <Bar dataKey="teamSizeRange" yAxisId="team" stackId="team" name="teamSizeRange" barSize={28}>
          {chartData.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.fits ? "var(--chart-1)" : "var(--muted)"}
              fillOpacity={0.4}
            />
          ))}
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
  );
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
              <span className="font-medium text-foreground">{point.teamSizeIde.toFixed(1)} devs</span>
              <span>Team size (CLI):</span>
              <span className="font-medium text-foreground">{point.teamSizeCli.toFixed(1)} devs</span>
              <span>Team size (avg):</span>
              <span className="font-medium text-foreground">{point.teamSizeAvg.toFixed(1)} devs</span>
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
  const items = [
    { label: "CLI-workflow capacity", color: "var(--chart-1)", opacity: 1 },
    { label: "IDE-workflow capacity", color: "var(--chart-1)", opacity: 0.4 },
    { label: "Model doesn't fit", color: "var(--muted)", opacity: 1 },
    { label: "% of SOTA", color: "var(--chart-2)", opacity: 1, isLine: true },
  ];

  return (
    <div className="flex flex-wrap items-center justify-center gap-4 pt-3 text-xs">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-1.5">
          {item.isLine ? (
            <div className="h-0.5 w-4 rounded" style={{ backgroundColor: item.color }} />
          ) : (
            <div
              className="h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: item.color, opacity: item.opacity }}
            />
          )}
          <span className="text-muted-foreground">{item.label}</span>
        </div>
      ))}
    </div>
  );
}
