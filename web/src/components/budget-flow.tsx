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
import { ApiHostingChart } from "@/components/api-hosting-chart";
import { ChartSelector, type ChartTab } from "@/components/chart-selector";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useApiPricing } from "@/lib/data";

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
  gpus,
  models,
  benchmarks,
  sotaScores,
  benchmarkCategory,
  settings,
  currencySymbol = "$",
}: BudgetFlowProps) {
  const [gpuConfig, setGpuConfig] = useState<PresetGpuConfig>(GPU_PRESETS[0]);
  const [memoryUtilization, setMemoryUtilization] = useState(90);
  const [ideRequestsPerHour, setIdeRequestsPerHour] = useState(50);
  const [cliRequestsPerHour, setCliRequestsPerHour] = useState(200);
  const [configExpanded, setConfigExpanded] = useState(false);

  const { pricing: closedPricing, loading: apiPricingLoading } = useApiPricing();

  const chartData = useMemo(
    () =>
      calculateBudgetChartData(
        gpuConfig,
        models,
        benchmarks,
        sotaScores,
        benchmarkCategory,
        memoryUtilization / 100,
        ideRequestsPerHour,
        cliRequestsPerHour,
        settings,
      ),
    [gpuConfig, models, benchmarks, sotaScores, benchmarkCategory, memoryUtilization, ideRequestsPerHour, cliRequestsPerHour, settings],
  );

  const fittingModels = useMemo(
    () =>
      chartData
        .filter((d) => d.fits)
        .map((d) => {
          const model = models.find((m) => m.model_name === d.modelName);
          return model ? { model, sotaPercent: d.percentOfSota } : null;
        })
        .filter((x): x is { model: Model; sotaPercent: number } => x !== null),
    [chartData, models],
  );

  const interconnectLabel = isNvLink(gpuConfig.interconnect) ? " NVLink" : "";

  const teamCapacityContent = (
    <>
      {/* IDE/CLI controls — moved inside Team Capacity tab */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
        <div className="space-y-2">
          <div className="flex items-center gap-1.5">
            <Label>IDE requests/hour</Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground cursor-help text-xs">&#9432;</span>
                </TooltipTrigger>
                <TooltipContent>
                  Average LLM requests per hour for IDE-workflow developers (e.g. Cursor completions, inline edits).
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <StepperInput
            value={ideRequestsPerHour}
            onChange={setIdeRequestsPerHour}
            min={10}
            max={200}
            step={10}
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-1.5">
            <Label>CLI requests/hour</Label>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground cursor-help text-xs">&#9432;</span>
                </TooltipTrigger>
                <TooltipContent>
                  Average LLM requests per hour for CLI-workflow developers (e.g. Claude Code agentic tasks).
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <StepperInput
            value={cliRequestsPerHour}
            onChange={setCliRequestsPerHour}
            min={50}
            max={500}
            step={50}
          />
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Development Team Capacity</CardTitle>
          <CardDescription>
            Number of developers each model can serve on {gpuConfig.label}{interconnectLabel}, at {memoryUtilization}% GPU memory utilization.
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
    </>
  );

  const apiVsSelfHostingContent = apiPricingLoading ? (
    <div className="h-[300px] flex items-center justify-center text-muted-foreground text-sm">
      Loading API pricing data...
    </div>
  ) : (
    <ApiHostingChart
      closedPricing={closedPricing}
      availableModels={fittingModels}
      gpuConfig={gpuConfig}
      gpus={gpus}
      memoryUtilization={memoryUtilization / 100}
      settings={settings}
      benchmarks={benchmarks}
      benchmarkCategory={benchmarkCategory}
      currencySymbol={currencySymbol}
    />
  );

  return (
    <div className="space-y-6">
      {/* GPU setup — collapsible, shared across charts */}
      <div className="rounded-lg border">
        <button
          onClick={() => setConfigExpanded(!configExpanded)}
          className="flex items-center justify-between w-full text-left text-sm px-3 py-2.5 hover:bg-accent/50 transition-colors cursor-pointer rounded-lg"
        >
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-medium truncate">{gpuConfig.label}{interconnectLabel}</span>
            <span className="text-muted-foreground shrink-0">&middot;</span>
            <span className="text-muted-foreground shrink-0">memory utilization: {memoryUtilization}%</span>
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

        {configExpanded && (
          <div className="px-3 pb-3 pt-1 space-y-6 border-t">
            <GpuConfigSelector value={gpuConfig} onChange={setGpuConfig} />

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>GPU Memory Utilization</Label>
                  <span className="text-sm font-medium tabular-nums">{memoryUtilization}%</span>
                </div>
                <Slider
                  value={[memoryUtilization]}
                  onValueChange={([v]) => setMemoryUtilization(v)}
                  min={70}
                  max={95}
                  step={5}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <ChartSelector
        tabs={[
          {
            value: "team-capacity",
            label: "Team Capacity",
            content: teamCapacityContent,
          },
          {
            value: "api-vs-self-hosting",
            label: "API vs. Self-Hosting",
            content: apiVsSelfHostingContent,
          },
        ] satisfies ChartTab[]}
      />
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
