"use client";

import { useMemo } from "react";
import type {
  GpuOffering,
  Model,
  BenchmarkScore,
  SotaScore,
  AdvancedSettings,
  PresetGpuConfig,
} from "@/types";
import { calculateBudgetMatrix } from "@/lib/matrix-calculator";
import { GpuConfigSelector } from "@/components/gpu-config-selector";
import { RecommendationMatrix } from "@/components/recommendation-matrix";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

interface BudgetFlowProps {
  gpus: GpuOffering[];
  models: Model[];
  benchmarks: BenchmarkScore[];
  sotaScores: SotaScore[];
  benchmarkCategory: string;
  gpuConfig: PresetGpuConfig | null;
  onGpuConfigChange: (config: PresetGpuConfig) => void;
  settings: AdvancedSettings;
  currencySymbol?: string;
}

export function BudgetFlow({
  gpus,
  models,
  benchmarks,
  sotaScores,
  benchmarkCategory,
  gpuConfig,
  onGpuConfigChange,
  settings,
  currencySymbol = "$",
}: BudgetFlowProps) {
  const matrix = useMemo(() => {
    if (!gpuConfig) return [];
    return calculateBudgetMatrix(
      gpuConfig,
      gpus,
      models,
      benchmarks,
      sotaScores,
      benchmarkCategory,
      settings,
    );
  }, [gpuConfig, gpus, models, benchmarks, sotaScores, benchmarkCategory, settings]);

  return (
    <div className="space-y-6">
      <GpuConfigSelector value={gpuConfig} onChange={onGpuConfigChange} />

      {gpuConfig ? (
        <Card>
          <CardHeader>
            <CardTitle>Best Models for {gpuConfig.label}</CardTitle>
            <CardDescription>
              Top-performing models that fit your GPU configuration, with cost per stream
              at each concurrency level.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {matrix.length > 0 ? (
              <RecommendationMatrix rows={matrix} persona="budget" currencySymbol={currencySymbol} />
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No models fit this GPU configuration at their native precision.
                Try a configuration with more VRAM.
              </div>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="text-center py-8 text-muted-foreground">
          Select a GPU configuration above to see recommendations.
        </div>
      )}
    </div>
  );
}
