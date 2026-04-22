"use client";

import { useMemo, useState } from "react";
import type { Model, GpuOffering, AdvancedSettings } from "@/types";
import { GapChart } from "@/components/gap-chart";
import { CostTrendChart } from "@/components/cost-trend-chart";
import { SotaPercentChart } from "@/components/sota-percent-chart";
import { ModelSizeChart } from "@/components/model-size-chart";
import { EfficiencyChart } from "@/components/efficiency-chart";
import { ScalingChart } from "@/components/scaling-chart";
import { ApiHostingChart } from "@/components/api-hosting-chart";
import { ChartSelector, type ChartTab } from "@/components/chart-selector";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  useSnapshotData,
  computeGapTrend,
  computeCostTrend,
  computeSotaPercentTrend,
  computeModelSizeScore,
  computeEfficiencyTrend,
  computeScalingCurve,
  computeGpuReferenceCosts,
  findBestOpenSourceModel,
  resolveModelName,
} from "@/lib/trend-data";
import { useApiPricing } from "@/lib/data";

interface TrendsSectionProps {
  models: Model[];
  gpus: GpuOffering[];
  benchmarkCategory: string;
  settings: AdvancedSettings;
  benchmarks: import("@/types").BenchmarkScore[];
  currencySymbol?: string;
}

export function TrendsSection({
  models,
  gpus,
  benchmarkCategory,
  settings,
  benchmarks,
  currencySymbol = "$",
}: TrendsSectionProps) {
  const { snapshots, loading } = useSnapshotData();
  const { pricing: apiPricing, loading: apiPricingLoading } = useApiPricing();

  // Set of open-source model names (from models.json)
  const openSourceNames = useMemo(
    () => new Set(models.map((m) => m.model_name)),
    [models],
  );

  // Chart 1: Gap Trend
  const gapData = useMemo(
    () =>
      snapshots.length > 0
        ? computeGapTrend(snapshots, openSourceNames, benchmarkCategory)
        : [],
    [snapshots, openSourceNames, benchmarkCategory],
  );

  // Chart 2: Cost Trend (derived from gap trend points)
  const costData = useMemo(
    () =>
      gapData.length > 0
        ? computeCostTrend(gapData, models, gpus, settings)
        : [],
    [gapData, models, gpus, settings],
  );

  // Chart 3: SOTA percent trend (derived from gap trend points)
  const sotaPercentData = useMemo(
    () => computeSotaPercentTrend(gapData),
    [gapData],
  );

  // Chart 4: Model size vs. score scatter (uses live benchmarks, not snapshots)
  const modelSizeData = useMemo(
    () => computeModelSizeScore(benchmarks, openSourceNames, models, benchmarkCategory),
    [benchmarks, openSourceNames, models, benchmarkCategory],
  );

  // Chart 5: Efficiency Trend (API cost per task for best models)
  const efficiencyData = useMemo(
    () =>
      gapData.length > 0 && snapshots.length > 0
        ? computeEfficiencyTrend(gapData, snapshots, benchmarkCategory)
        : [],
    [gapData, snapshots, benchmarkCategory],
  );

  // GPU reference costs (shared by Charts 2 & 4)
  const referenceCosts = useMemo(
    () => computeGpuReferenceCosts(gpus),
    [gpus],
  );

  // Chart 3: Scaling Curve — best model auto-selected, user can override
  const bestModel = useMemo(
    () =>
      findBestOpenSourceModel(benchmarks, openSourceNames, models, benchmarkCategory),
    [benchmarks, openSourceNames, models, benchmarkCategory],
  );

  // All open-source models with scores in the selected category, sorted best first
  const availableScalingModels = useMemo(() => {
    const scoreMap = new Map<string, number>();
    for (const b of benchmarks) {
      if (b.benchmark_name === benchmarkCategory && b.score !== null) {
        const resolved = resolveModelName(b.model_name);
        if (!openSourceNames.has(resolved)) continue;
        const prev = scoreMap.get(resolved);
        if (prev === undefined || b.score! > prev) scoreMap.set(resolved, b.score!);
      }
    }
    return [...scoreMap.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([name]) => models.find((m) => m.model_name === name))
      .filter((m): m is Model => m !== undefined);
  }, [benchmarks, benchmarkCategory, openSourceNames, models]);

  const [selectedModelName, setSelectedModelName] = useState<string | null>(null);

  // Use the user's selection if it's still valid for this category, else fall back to best
  const scalingModel = useMemo(() => {
    if (selectedModelName) {
      const found = availableScalingModels.find((m) => m.model_name === selectedModelName);
      if (found) return found;
    }
    return bestModel;
  }, [selectedModelName, availableScalingModels, bestModel]);

  const scalingData = useMemo(
    () =>
      scalingModel
        ? computeScalingCurve(scalingModel, gpus, settings)
        : [],
    [scalingModel, gpus, settings],
  );

  const categoryDisplayName =
    benchmarks.find((b) => b.benchmark_name === benchmarkCategory)
      ?.benchmark_display_name ?? benchmarkCategory;

  if (loading) {
    return (
      <section className="mt-16">
        <h2 className="text-2xl font-bold tracking-tight mb-1">Trends</h2>
        <p className="text-muted-foreground mb-8">
          Loading historical data...
        </p>
      </section>
    );
  }

  return (
    <section className="mt-16">
      <div className="mb-6">
        <h2 className="text-2xl font-bold tracking-tight mb-1">Trends</h2>
        <p className="text-muted-foreground">
          Results are updated as the OpenHands Index leaderboard changes.
          Watch this space for updates.
        </p>
      </div>

      <ChartSelector
        tabs={[
          { value: "gap", label: "Open vs Closed Gap", content: <GapChart data={gapData} /> },
          {
            value: "api-vs-self-hosting",
            label: "API vs Self-hosting",
            content: apiPricingLoading ? (
              <div className="h-[300px] flex items-center justify-center text-muted-foreground text-sm">
                Loading API pricing data...
              </div>
            ) : (
              <ApiHostingChart
                closedPricing={apiPricing}
                availableModels={availableScalingModels}
                gpus={gpus}
                settings={settings}
                currencySymbol={currencySymbol}
              />
            ),
          },
          { value: "cost", label: "Cost Trend", content: <CostTrendChart data={costData} referenceCosts={referenceCosts} currencySymbol={currencySymbol} /> },
          { value: "sota", label: "% of SOTA", content: <SotaPercentChart data={sotaPercentData} /> },
          { value: "size", label: "Model Size", content: <ModelSizeChart data={modelSizeData} categoryDisplayName={categoryDisplayName} /> },
          { value: "efficiency", label: "Efficiency", content: <EfficiencyChart data={efficiencyData} categoryDisplayName={categoryDisplayName} /> },
          {
            value: "scaling",
            label: "Scaling Cost",
            content: (
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <label className="text-sm font-medium text-muted-foreground whitespace-nowrap">
                    Model
                  </label>
                  <Select
                    value={scalingModel?.model_name ?? ""}
                    onValueChange={(v) => setSelectedModelName(v)}
                  >
                    <SelectTrigger className="w-full sm:w-[320px]">
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableScalingModels.map((m, i) => (
                        <SelectItem key={m.model_name} value={m.model_name}>
                          {m.model_name}{i === 0 ? " (best)" : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <ScalingChart data={scalingData} referenceCosts={referenceCosts} modelName={scalingModel?.model_name ?? "N/A"} categoryDisplayName={categoryDisplayName} currencySymbol={currencySymbol} />
              </div>
            ),
          },
        ] satisfies ChartTab[]}
      />
    </section>
  );
}
