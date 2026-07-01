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
import { buildGpuPresets } from "@/lib/gpu-presets";
import { interconnectBadgeLabel } from "@/components/deployment-estimate-panel";
import { GpuConfigSelector } from "@/components/gpu-config-selector";
import { BudgetChart } from "@/components/budget-chart";
import { ApiHostingChart } from "@/components/api-hosting-chart";
import { ChartSelector, type ChartTab } from "@/components/chart-selector";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { useApiPricing } from "@/lib/data";

interface BudgetFlowProps {
  gpus: GpuOffering[];
  models: Model[];
  benchmarks: BenchmarkScore[];
  sotaScores: SotaScore[];
  benchmarkCategory: string;
  settings: AdvancedSettings;
  currencySymbol?: string;
  location?: string;
}

export function BudgetFlow({
  gpus,
  models,
  benchmarks,
  sotaScores,
  benchmarkCategory,
  settings,
  currencySymbol = "$",
  location,
}: BudgetFlowProps) {
  const gpuPresets = useMemo(() => buildGpuPresets(gpus, models, benchmarks), [gpus, models, benchmarks]);
  const [gpuConfig, setGpuConfig] = useState<PresetGpuConfig | null>(() => gpuPresets[0] ?? null);
  const [memoryUtilization, setMemoryUtilization] = useState(90);
  const [configExpanded, setConfigExpanded] = useState(false);

  // When the available presets change (e.g. the user switches region), keep the
  // current selection if it's still offered, otherwise fall back to the new
  // region's default. Adjusting state during render avoids a stale config — and
  // the resulting render flicker of an effect — when switching into a region
  // whose catalog can't host the previously selected GPU.
  const [prevPresets, setPrevPresets] = useState(gpuPresets);
  if (prevPresets !== gpuPresets) {
    setPrevPresets(gpuPresets);
    const stillOffered = gpuConfig !== null && gpuPresets.some((p) => p.label === gpuConfig.label);
    if (!stillOffered) {
      setGpuConfig(gpuPresets[0] ?? null);
    }
  }

  const { pricing: closedPricing, loading: apiPricingLoading } = useApiPricing();

  const chartData = useMemo(
    () =>
      gpuConfig
        ? calculateBudgetChartData(
            gpuConfig,
            models,
            benchmarks,
            sotaScores,
            benchmarkCategory,
            memoryUtilization / 100,
            settings,
          )
        : [],
    [gpuConfig, models, benchmarks, sotaScores, benchmarkCategory, memoryUtilization, settings],
  );

  const fittingModels = useMemo(
    () =>
      chartData
        // Unranked models have no benchmark score, so there is no cost-vs-API
        // comparison to make — exclude them from the API-vs-Self-Hosting chart.
        // Serving Capacity (BudgetChart) still shows them.
        .filter((d) => d.fits && !d.isUnranked)
        .map((d) => {
          const model = models.find((m) => m.model_name === d.modelName);
          if (!model || d.percentOfSota === null) return null;
          return { model, sotaPercent: d.percentOfSota };
        })
        .filter((x): x is { model: Model; sotaPercent: number } => x !== null),
    [chartData, models],
  );

  // No preset GPU config can host the top models in this region (e.g. a region
  // whose catalog only has small-VRAM pods). Short-circuit instead of crashing.
  if (!gpuConfig) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Serving Capacity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            No GPU configuration{location ? ` in ${location}` : ""} can host the
            current top models. Try a different region.
          </div>
        </CardContent>
      </Card>
    );
  }

  const interconnectLabel = ` ${interconnectBadgeLabel(gpuConfig.gpuName, gpuConfig.gpuCount)}`;

  const teamCapacityContent = (
    <Card>
      <CardHeader>
        <CardTitle>Serving Capacity</CardTitle>
        <CardDescription>
          Request throughput and team capacity for each model on {gpuConfig.label}{interconnectLabel}, at {memoryUtilization}% GPU memory utilization.
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
      <div className="space-y-1.5">
        <Label className="text-sm font-medium text-muted-foreground">GPU configuration</Label>
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
            <GpuConfigSelector value={gpuConfig} onChange={setGpuConfig} presets={gpuPresets} />

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
      </div>

      <ChartSelector
        tabs={[
          {
            value: "serving-capacity",
            label: "Serving Capacity",
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

