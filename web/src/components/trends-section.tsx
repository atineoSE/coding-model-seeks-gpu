"use client";

import { useMemo } from "react";
import type { Model, GpuOffering, AdvancedSettings } from "@/types";
import { GapChart } from "@/components/gap-chart";
import { CostTrendChart } from "@/components/cost-trend-chart";
import { ScalingChart } from "@/components/scaling-chart";
import {
  useSnapshotData,
  computeGapTrend,
  computeCostTrend,
  computeScalingCurve,
  computeGpuReferenceCosts,
  findBestOpenSourceModel,
} from "@/lib/trend-data";

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

  // Chart 2: Cost Trend
  const costData = useMemo(
    () =>
      snapshots.length > 0
        ? computeCostTrend(
            snapshots,
            openSourceNames,
            models,
            gpus,
            benchmarkCategory,
            settings,
          )
        : [],
    [snapshots, openSourceNames, models, gpus, benchmarkCategory, settings],
  );

  // GPU reference costs (shared by Charts 2 & 3)
  const referenceCosts = useMemo(
    () => computeGpuReferenceCosts(gpus),
    [gpus],
  );

  // Chart 3: Scaling Curve (uses latest snapshot's best open-source model)
  const bestModel = useMemo(
    () =>
      findBestOpenSourceModel(benchmarks, openSourceNames, models, benchmarkCategory),
    [benchmarks, openSourceNames, models, benchmarkCategory],
  );

  const scalingData = useMemo(
    () =>
      bestModel
        ? computeScalingCurve(bestModel, gpus, settings)
        : [],
    [bestModel, gpus, settings],
  );

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

      <div className="grid gap-6">
        <GapChart data={gapData} />
        <CostTrendChart data={costData} referenceCosts={referenceCosts} currencySymbol={currencySymbol} />
        <ScalingChart
          data={scalingData}
          referenceCosts={referenceCosts}
          modelName={bestModel?.model_name ?? "N/A"}
          categoryDisplayName={
            benchmarks.find((b) => b.benchmark_name === benchmarkCategory)
              ?.benchmark_display_name ?? benchmarkCategory
          }
          currencySymbol={currencySymbol}
        />
      </div>
    </section>
  );
}
