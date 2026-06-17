"use client";

import { useMemo } from "react";
import type { GpuOffering, Model, BenchmarkScore, SotaScore, AdvancedSettings } from "@/types";
import { calculatePerformanceMatrix, calculateUnrankedMatrix } from "@/lib/matrix-calculator";
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

  const unrankedMatrix = useMemo(
    () => calculateUnrankedMatrix(gpus, models, benchmarks, benchmarkCategory, settings),
    [gpus, models, benchmarks, benchmarkCategory, settings],
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

      {unrankedMatrix.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Unranked Models</CardTitle>
            <CardDescription>
              Open models with known size but no OpenHands Index result yet, so they can&apos;t
              be ranked above. Listed biggest first by VRAM, with the cheapest GPU setup per
              concurrency level and links to their license and HuggingFace page.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <RecommendationMatrix
              rows={unrankedMatrix}
              persona="performance"
              currencySymbol={currencySymbol}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
