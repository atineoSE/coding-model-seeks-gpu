"use client";

import { useState, useMemo } from "react";
import type { Persona, AdvancedSettings, PresetGpuConfig } from "@/types";
import { DEFAULT_ADVANCED_SETTINGS } from "@/lib/matrix-calculator";
import { useData, getBenchmarkGroups, getLocations, deduplicateGpus } from "@/lib/data";
import { getModelMemory, resolveModelPrecision, WEIGHT_OVERHEAD_FACTOR } from "@/lib/calculations";
import { getGpuThroughputSpec } from "@/lib/gpu-specs";
import { ThemeToggle } from "@/components/theme-toggle";
import { RegionFilter } from "@/components/region-filter";
import { CategorySelector, type BenchmarkCategory } from "@/components/category-selector";

import { PersonaSelector } from "@/components/persona-selector";
import { PerformanceFlow } from "@/components/performance-flow";
import { BudgetFlow } from "@/components/budget-flow";
import { AdvancedSettingsDialog } from "@/components/advanced-settings-dialog";
import { TrendsSection } from "@/components/trends-section";

const ENABLE_LOCATION_FILTER =
  process.env.NEXT_PUBLIC_ENABLE_LOCATION_FILTER === "true";

export default function Home() {
  const { data, loading } = useData();

  const benchmarkGroups = useMemo(
    () => (data ? getBenchmarkGroups(data.benchmarks) : []),
    [data],
  );

  // Derive available categories from benchmark data
  const categories: BenchmarkCategory[] = useMemo(() => {
    if (benchmarkGroups.length === 0) return [];
    const group = benchmarkGroups[0];
    if (!group) return [];
    return group.types.map((t) => ({
      name: t.name,
      displayName: t.displayName,
    }));
  }, [benchmarkGroups]);

  const [persona, setPersona] = useState<Persona>("performance");
  const [benchmarkCategory, setBenchmarkCategory] = useState("");
  const [gpuConfig, setGpuConfig] = useState<PresetGpuConfig | null>(null);
  const [settings, setSettings] = useState<AdvancedSettings>(DEFAULT_ADVANCED_SETTINGS);
  const [location, setLocation] = useState("");

  const locations = useMemo(() => {
    if (!data) return [];
    // Find the smallest model memory across all models
    const minModelMem = Math.min(
      ...data.models
        .map((m) => getModelMemory(m, resolveModelPrecision(m)))
        .filter((v): v is number => v !== null),
    );
    if (!isFinite(minModelMem)) return getLocations(data.gpus);
    const minVram = minModelMem * WEIGHT_OVERHEAD_FACTOR;
    // Keep only locations that have at least one recognized GPU with enough VRAM
    return getLocations(data.gpus).filter((loc) => {
      const locGpus = data.gpus.filter((g) => g.location === loc);
      return locGpus.some(
        (g) => getGpuThroughputSpec(g.gpu_name) !== null && g.total_vram_gb >= minVram,
      );
    });
  }, [data]);

  // Default to location with most offerings once data loads
  if (!location && locations.length > 0) {
    setLocation(locations[0]);
  }

  const filteredGpus = useMemo(() => {
    if (!data) return [];
    if (ENABLE_LOCATION_FILTER) {
      return location ? data.gpus.filter((g) => g.location === location) : [];
    }
    return deduplicateGpus(data.gpus);
  }, [data, location]);

  // Set defaults once data loads
  const defaultsSet = useMemo(() => {
    if (!data || benchmarkCategory) return true;
    return false;
  }, [data, benchmarkCategory]);

  if (!defaultsSet && categories.length > 0) {
    const overall = categories.find((c) => c.name.includes("overall"));
    setBenchmarkCategory(overall?.name ?? categories[0].name);
  }

  if (loading) {
    return (
      <main className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-muted-foreground">Loading data...</p>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 pb-8 max-w-6xl">
        {/* Sticky header */}
        <div className="sticky top-0 z-10 bg-background pt-6 pb-4">
          {/* Title row */}
          <div className="flex items-start justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">
                Coding Model Seeks GPU
                <span className="ml-2 text-sm font-normal text-muted-foreground align-middle">
                  v0.1
                </span>
              </h1>
              <p className="text-muted-foreground mt-1">
                Open source coding LLMs ranked by real-world performance, sized to real hardware.
                {data?.updatedAt && (
                  <span className="ml-2 text-xs text-muted-foreground/60">
                    Updated {new Date(data.updatedAt).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })}
                  </span>
                )}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <AdvancedSettingsDialog
                settings={settings}
                onSettingsChange={setSettings}
              />
              <a
                href="https://github.com/atineoSE/coding-model-seeks-gpu"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center h-9 w-9 rounded-md text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
                aria-label="View on GitHub"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12Z"/></svg>
              </a>
              <ThemeToggle />
            </div>
          </div>

          {/* Filters row — stack on mobile */}
          <div className="flex flex-col sm:flex-row items-start gap-4 mb-4">
            {ENABLE_LOCATION_FILTER && (
              <RegionFilter
                locations={locations}
                value={location}
                onChange={setLocation}
              />
            )}
            <CategorySelector
              categories={categories}
              value={benchmarkCategory}
              onChange={setBenchmarkCategory}
            />
          </div>
        </div>

        {/* Description — scrolls away, not sticky */}
        <p className="text-sm text-muted-foreground mb-4">
          Models are ranked using the{" "}
          <a
            href="https://index.openhands.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-foreground transition-colors"
          >
            OpenHands Index
          </a>
          , a benchmark that evaluates open source coding LLMs on real-world software engineering tasks. For each model, we calculate the VRAM requirements and NVIDIA GPU configurations needed to run it at different throughput levels. Use it to find the right hardware for your model or the best model that fits your hardware.
        </p>

        {/* Persona selector */}
        <div className="mb-8">
          <PersonaSelector value={persona} onChange={setPersona} />
        </div>

        {/* Flow content */}
        {data && persona === "performance" && benchmarkCategory && (
          <div className="mt-8">
            <PerformanceFlow
              gpus={filteredGpus}
              models={data.models}
              benchmarks={data.benchmarks}
              sotaScores={data.sotaScores}
              benchmarkCategory={benchmarkCategory}
              settings={settings}
              currencySymbol={data.gpuSource.currency_symbol}
            />
          </div>
        )}

        {data && persona === "budget" && benchmarkCategory && (
          <div className="mt-8">
            <BudgetFlow
              gpus={filteredGpus}
              models={data.models}
              benchmarks={data.benchmarks}
              sotaScores={data.sotaScores}
              benchmarkCategory={benchmarkCategory}
              gpuConfig={gpuConfig}
              onGpuConfigChange={setGpuConfig}
              settings={settings}
              currencySymbol={data.gpuSource.currency_symbol}
            />
          </div>
        )}

        {data && persona === "trends" && (
          <div className="mt-8">
            <TrendsSection
              models={data.models}
              gpus={filteredGpus}
              benchmarks={data.benchmarks}
              benchmarkCategory={benchmarkCategory}
              settings={settings}
              currencySymbol={data.gpuSource.currency_symbol}
            />
          </div>
        )}

        {/* Disclaimer */}
        <footer className="mt-12 text-xs text-muted-foreground/60 text-center max-w-2xl mx-auto space-y-2">
          {data && persona !== "budget" && (
            <p>
              GPU pricing from{" "}
              <a
                href={data.gpuSource.service_url}
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-muted-foreground transition-colors"
              >
                {data.gpuSource.service_name}
              </a>
              . {data.gpuSource.description}
            </p>
          )}
          <p>
            All calculations are estimates based on theoretical models and publicly available specifications. Results should be considered approximations. Actual performance may vary — benchmark on real hardware with your target workloads for precise data.
          </p>
        </footer>
      </div>
    </main>
  );
}
