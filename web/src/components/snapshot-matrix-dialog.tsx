"use client";

import { useState } from "react";
import { Grid2X2Check } from "lucide-react";
import type { BenchmarkScore, Model } from "@/types";
import {
  getMatrixModels,
  BENCHMARK_CATEGORIES,
  CATEGORY_DISPLAY_NAMES,
  type BenchmarkCategory,
} from "@/lib/snapshot-matrix";
import { useApiPricing } from "@/lib/data";
import { Button } from "@/components/ui/button";
import { formatModelName, formatLabName } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface SnapshotMatrixDialogProps {
  benchmarks: BenchmarkScore[];
  models: Model[];
}

type MatrixModel = ReturnType<typeof getMatrixModels>[number];

function MatrixSection({
  title,
  models,
  colSpan,
}: {
  title: string;
  models: MatrixModel[];
  colSpan: number;
}) {
  if (models.length === 0) return null;
  return (
    <tbody>
      <tr>
        <td
          colSpan={colSpan}
          className="pt-5 pb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground"
        >
          {title}
        </td>
      </tr>
      {models.map((model) => (
        <ModelRow key={model.modelName} model={model} />
      ))}
    </tbody>
  );
}

function ModelRow({ model }: { model: MatrixModel }) {
  return (
    <tr className="border-b last:border-b-0">
      <td className="py-2 pr-4 font-medium whitespace-nowrap sticky left-0 bg-background">
        {formatModelName(model.modelName)}
        {model.lab && (
          <span className="ml-1.5 text-xs text-muted-foreground">
            ({formatLabName(model.lab)})
          </span>
        )}
      </td>
      {BENCHMARK_CATEGORIES.map((cat: BenchmarkCategory) => {
        const score = model.scores[cat];
        const hasScore = score != null;
        return (
          <td
            key={cat}
            className={`text-center py-2 px-2 tabular-nums ${
              hasScore
                ? "text-emerald-600 dark:text-emerald-400"
                : "text-muted-foreground/40"
            }`}
          >
            {hasScore ? score.toFixed(1) : "—"}
          </td>
        );
      })}
    </tr>
  );
}

export function SnapshotMatrixDialog({ benchmarks, models }: SnapshotMatrixDialogProps) {
  const [open, setOpen] = useState(false);
  // Which closed model represents each lab is the pipeline's decision, published in
  // api_pricing.json — the same rows that drive the API-vs-self-hosting chart. Reading
  // it here keeps the two views in agreement instead of re-deriving the choice.
  const { pricing } = useApiPricing();
  const closedModelsPerLab = Object.fromEntries(pricing.map((p) => [p.lab, p.model_name]));
  const allModels = open ? getMatrixModels(benchmarks, models, closedModelsPerLab) : [];
  const closedModels = allModels.filter((m) => m.lab !== null);
  const rankedOpenModels = allModels.filter((m) => m.lab === null && !m.unranked);
  const unrankedOpenModels = allModels.filter((m) => m.lab === null && m.unranked);
  const colSpan = BENCHMARK_CATEGORIES.length + 1;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon-sm" title="Snapshot Coverage Matrix">
          <Grid2X2Check className="size-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Snapshot Coverage Matrix</DialogTitle>
          <DialogDescription>
            Benchmark scores for each model in the latest snapshot. Gaps indicate missing data;
            unranked open-weights models are sized but have no OpenHands Index result yet.
          </DialogDescription>
        </DialogHeader>
        <div className="overflow-auto flex-1 -mx-6 px-6">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b">
                <th className="text-left py-2 pr-4 font-medium sticky left-0 bg-background">
                  Model
                </th>
                {BENCHMARK_CATEGORIES.map((cat) => (
                  <th
                    key={cat}
                    className="text-center py-2 px-2 font-medium whitespace-nowrap"
                  >
                    {CATEGORY_DISPLAY_NAMES[cat]}
                  </th>
                ))}
              </tr>
            </thead>

            {allModels.length === 0 ? (
              <tbody>
                <tr>
                  <td colSpan={colSpan} className="text-center py-8 text-muted-foreground">
                    No snapshot data available.
                  </td>
                </tr>
              </tbody>
            ) : (
              <>
                <MatrixSection
                  title="Leading Closed API Models"
                  models={closedModels}
                  colSpan={colSpan}
                />
                <MatrixSection
                  title="Ranked Open Weights Models"
                  models={rankedOpenModels}
                  colSpan={colSpan}
                />
                <MatrixSection
                  title="Unranked Open Weights Models"
                  models={unrankedOpenModels}
                  colSpan={colSpan}
                />
              </>
            )}
          </table>
        </div>
      </DialogContent>
    </Dialog>
  );
}
