"use client";

import { useState, useMemo } from "react";
import type {
  GpuOffering,
  Model,
  BenchmarkScore,
  SotaScore,
  AdvancedSettings,
  PresetGpuConfig,
} from "@/types";
import { calculateBudgetChartData } from "@/lib/matrix-calculator";
import { GPU_PRESETS } from "@/lib/gpu-presets";
import { isNvLink } from "@/lib/calculations";
import { GpuConfigSelector } from "@/components/gpu-config-selector";
import { BudgetChart } from "@/components/budget-chart";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface BudgetFlowProps {
  gpus: GpuOffering[];
  models: Model[];
  benchmarks: BenchmarkScore[];
  sotaScores: SotaScore[];
  benchmarkCategory: string;
  settings: AdvancedSettings;
  currencySymbol?: string;
}

export function BudgetFlow({
  models,
  benchmarks,
  sotaScores,
  benchmarkCategory,
  settings,
}: BudgetFlowProps) {
  // Internal state — no longer lifted to page.tsx
  const [gpuConfig, setGpuConfig] = useState<PresetGpuConfig>(GPU_PRESETS[0]);
  const [targetUtilization, setTargetUtilization] = useState(80);
  const [minTokPerSec, setMinTokPerSec] = useState(100);
  const [ideStreamsPerDev, setIdeStreamsPerDev] = useState(1.5);
  const [cliStreamsPerDev, setCliStreamsPerDev] = useState(4);
  const [configExpanded, setConfigExpanded] = useState(false);

  const chartData = useMemo(
    () =>
      calculateBudgetChartData(
        gpuConfig,
        models,
        benchmarks,
        sotaScores,
        benchmarkCategory,
        targetUtilization / 100,
        minTokPerSec,
        ideStreamsPerDev,
        cliStreamsPerDev,
        settings,
      ),
    [gpuConfig, models, benchmarks, sotaScores, benchmarkCategory, targetUtilization, minTokPerSec, ideStreamsPerDev, cliStreamsPerDev, settings],
  );

  const interconnectLabel = isNvLink(gpuConfig.interconnect) ? " NVLink" : "";

  return (
    <div className="space-y-6">
      {/* GPU setup — collapsible, shared across charts */}
      <div className="rounded-lg border">
        {/* Summary bar — always visible */}
        <button
          onClick={() => setConfigExpanded(!configExpanded)}
          className="flex items-center justify-between w-full text-left text-sm px-3 py-2.5 hover:bg-accent/50 transition-colors cursor-pointer rounded-lg"
        >
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-medium truncate">{gpuConfig.label}{interconnectLabel}</span>
            <span className="text-muted-foreground shrink-0">&middot;</span>
            <span className="text-muted-foreground shrink-0">{targetUtilization}% util</span>
            <span className="text-muted-foreground shrink-0">&middot;</span>
            <span className="text-muted-foreground shrink-0">{minTokPerSec} tok/s</span>
          </div>
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            className="shrink-0 ml-2 text-muted-foreground"
          >
            {configExpanded ? (
              <path d="M4 10L8 6L12 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            )}
          </svg>
        </button>

        {/* Expanded configuration */}
        {configExpanded && (
          <div className="px-3 pb-3 pt-1 space-y-6 border-t">
            <GpuConfigSelector value={gpuConfig} onChange={setGpuConfig} />

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Utilization slider */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Target Utilization</Label>
                  <span className="text-sm font-medium tabular-nums">{targetUtilization}%</span>
                </div>
                <Slider
                  value={[targetUtilization]}
                  onValueChange={([v]) => setTargetUtilization(v)}
                  min={50}
                  max={90}
                  step={5}
                />
              </div>

              {/* Min tok/s slider */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>Min Decode Throughput</Label>
                  <span className="text-sm font-medium tabular-nums">{minTokPerSec} tok/s</span>
                </div>
                <Slider
                  value={[minTokPerSec]}
                  onValueChange={([v]) => setMinTokPerSec(v)}
                  min={50}
                  max={200}
                  step={10}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* IDE-workflow streams/dev */}
              <div className="space-y-2">
                <div className="flex items-center gap-1.5">
                  <Label>IDE-workflow streams/dev</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="text-muted-foreground cursor-help text-xs">&#9432;</span>
                      </TooltipTrigger>
                      <TooltipContent>
                        IDE-workflow is representative of development activities focused around the IDE and short-horizon tasks like Cursor.
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <StepperInput
                  value={ideStreamsPerDev}
                  onChange={setIdeStreamsPerDev}
                  min={0.5}
                  max={10}
                  step={0.5}
                />
              </div>

              {/* CLI-workflow streams/dev */}
              <div className="space-y-2">
                <div className="flex items-center gap-1.5">
                  <Label>CLI-workflow streams/dev</Label>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="text-muted-foreground cursor-help text-xs">&#9432;</span>
                      </TooltipTrigger>
                      <TooltipContent>
                        CLI-workflow focuses on terminal and long-horizon tasks like Claude Code.
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <StepperInput
                  value={cliStreamsPerDev}
                  onChange={setCliStreamsPerDev}
                  min={1}
                  max={20}
                  step={1}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Team capacity chart */}
      <Card>
        <CardHeader>
          <CardTitle>Development Team Capacity</CardTitle>
          <CardDescription>
            Number of developers each model can serve on {gpuConfig.label}{interconnectLabel}, at {targetUtilization}% utilization.
            Bar shows average capacity; whiskers indicate CLI (lower) and IDE (upper) workflow range.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {chartData.length > 0 ? (
            <BudgetChart data={chartData} />
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              No models available for the selected benchmark category.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/** Number stepper with +/- buttons */
function StepperInput({
  value,
  onChange,
  min,
  max,
  step,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  const decrement = () => onChange(Math.max(min, +(value - step).toFixed(2)));
  const increment = () => onChange(Math.min(max, +(value + step).toFixed(2)));

  return (
    <div className="flex items-center gap-1">
      <button
        onClick={decrement}
        disabled={value <= min}
        className="flex items-center justify-center h-8 w-8 rounded-md border text-sm font-medium hover:bg-accent disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
      >
        &minus;
      </button>
      <span className="flex items-center justify-center h-8 w-14 rounded-md border bg-transparent text-sm font-medium tabular-nums">
        {value % 1 === 0 ? value : value.toFixed(1)}
      </span>
      <button
        onClick={increment}
        disabled={value >= max}
        className="flex items-center justify-center h-8 w-8 rounded-md border text-sm font-medium hover:bg-accent disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
      >
        +
      </button>
    </div>
  );
}
