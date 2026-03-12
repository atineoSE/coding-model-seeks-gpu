"use client";

import { useMemo } from "react";
import type { GpuOffering, Model, BenchmarkScore, SotaScore, AdvancedSettings } from "@/types";
import { calculatePerformanceMatrix } from "@/lib/matrix-calculator";
import { computeTotalBenchmarkCost } from "@/lib/benchmark-costs";
import { RecommendationMatrix } from "@/components/recommendation-matrix";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

interface PerformanceFlowProps {
  gpus: GpuOffering[];
  models: Model[];
  benchmarks: BenchmarkScore[];
  sotaScores: SotaScore[];
  benchmarkCategory: string;
  settings: AdvancedSettings;
  currencySymbol?: string;
}

export function PerformanceFlow({
  gpus,
  models,
  benchmarks,
  sotaScores,
  benchmarkCategory,
  settings,
  currencySymbol = "$",
}: PerformanceFlowProps) {
  const matrix = useMemo(
    () =>
      calculatePerformanceMatrix(
        gpus,
        models,
        benchmarks,
        sotaScores,
        benchmarkCategory,
        settings,
      ),
    [gpus, models, benchmarks, sotaScores, benchmarkCategory, settings],
  );

  const sota = sotaScores.find((s) => s.benchmark_name === benchmarkCategory) ?? null;

  const sotaTotalBenchmarkCost = useMemo(
    () =>
      sota
        ? computeTotalBenchmarkCost(sota.sota_model_name, benchmarkCategory, benchmarks)
        : null,
    [sota, benchmarkCategory, benchmarks],
  );

  const benchmarkDisplayName = useMemo(
    () =>
      benchmarks.find((b) => b.benchmark_name === benchmarkCategory)
        ?.benchmark_display_name ?? benchmarkCategory,
    [benchmarks, benchmarkCategory],
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Top Coding Models</CardTitle>
          <CardDescription>
            The best open-source models ranked by benchmark score, with the cheapest GPU
            setups that can serve each concurrency level.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <RecommendationMatrix
            rows={matrix}
            persona="performance"
            currencySymbol={currencySymbol}
            sotaTotalBenchmarkCost={sotaTotalBenchmarkCost}
            benchmarkDisplayName={benchmarkDisplayName}
          />
        </CardContent>
      </Card>
    </div>
  );
}
