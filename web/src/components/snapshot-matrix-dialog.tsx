"use client";

import { useState } from "react";
import { Grid2X2Check } from "lucide-react";
import type { BenchmarkScore } from "@/types";
import {
  getMatrixModels,
  BENCHMARK_CATEGORIES,
  CATEGORY_DISPLAY_NAMES,
  type BenchmarkCategory,
} from "@/lib/snapshot-matrix";
import { Button } from "@/components/ui/button";
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
}

export function SnapshotMatrixDialog({ benchmarks }: SnapshotMatrixDialogProps) {
  const [open, setOpen] = useState(false);
  const models = open ? getMatrixModels(benchmarks) : [];

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
            Benchmark scores for each model in the latest snapshot. Gaps indicate missing data.
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
            <tbody>
              {models.map((model) => (
                <tr key={model.modelName} className="border-b last:border-b-0">
                  <td className="py-2 pr-4 font-medium whitespace-nowrap sticky left-0 bg-background">
                    {model.modelName}
                    {model.lab && (
                      <span className="ml-1.5 text-xs text-muted-foreground">
                        ({model.lab.charAt(0).toUpperCase() + model.lab.slice(1)})
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
                        {hasScore ? score.toFixed(1) : "\u2014"}
                      </td>
                    );
                  })}
                </tr>
              ))}
              {models.length === 0 && (
                <tr>
                  <td
                    colSpan={BENCHMARK_CATEGORIES.length + 1}
                    className="text-center py-8 text-muted-foreground"
                  >
                    No snapshot data available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </DialogContent>
    </Dialog>
  );
}
