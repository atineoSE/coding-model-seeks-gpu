"use client";

import { useMemo } from "react";
import type { GpuOffering, Model, BenchmarkScore, AdvancedSettings } from "@/types";
import { calculateUnrankedModelRows, type UnrankedModelRow } from "@/lib/matrix-calculator";
import { CONCURRENCY_TIERS } from "@/lib/concurrency-tiers";
import { minVramForModel } from "@/lib/model-data";
import { isNvLink } from "@/lib/calculations";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatModelName } from "@/lib/utils";
import { ExternalLink } from "lucide-react";

interface UnrankedModelsSectionProps {
  gpus: GpuOffering[];
  models: Model[];
  benchmarks: BenchmarkScore[];
  benchmarkCategory: string;
  settings: AdvancedSettings;
  currencySymbol?: string;
}

function formatCurrency(value: number, symbol: string = "$"): string {
  if (value >= 1000) return `${symbol}${(value / 1000).toFixed(1)}k`;
  return `${symbol}${Math.round(value)}`;
}

function ModelHeader({ model }: { model: Model }) {
  const minVram = minVramForModel(model);
  const url = model.hf_model_id
    ? `https://huggingface.co/${model.hf_model_id}`
    : model.model_url ?? null;

  return (
    <div className="min-w-0">
      {/* Name — links to HuggingFace (or fallback URL) */}
      <div className="font-semibold text-sm">
        {url ? (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:underline inline-flex items-center gap-1"
          >
            {formatModelName(model.model_name)}
            <ExternalLink className="size-3 text-muted-foreground" />
          </a>
        ) : (
          formatModelName(model.model_name)
        )}
      </div>

      {/* Params + min VRAM + context */}
      <div className="text-xs text-muted-foreground mt-0.5">
        {model.learnable_params_b !== null && (
          <>
            {Math.round(model.learnable_params_b)}b
            {model.architecture === "MoE" && model.active_params_b !== null && (
              <> ({Math.round(model.active_params_b)}b active)</>
            )}
          </>
        )}
        {minVram !== null && (
          <>{model.learnable_params_b !== null ? " · " : ""}min {minVram} GB VRAM</>
        )}
        {model.context_length !== null && (
          <> · {Math.round(model.context_length / 1024)}K context</>
        )}
      </div>

      {/* Badges: unranked marker, MoE, precision, license */}
      <div className="flex flex-wrap items-center gap-1 mt-1.5">
        <Badge variant="secondary" className="text-[10px] text-muted-foreground">
          Unranked
        </Badge>
        {model.architecture === "MoE" && (
          <Badge variant="secondary" className="text-[10px]">
            MoE
          </Badge>
        )}
        {model.precision && (
          <Badge variant="secondary" className="text-[10px]">
            {model.precision}
          </Badge>
        )}
        {model.license_name &&
          (model.license_url ? (
            <a href={model.license_url} target="_blank" rel="noopener noreferrer">
              <Badge variant="outline" className="text-[10px] hover:bg-accent cursor-pointer">
                {model.license_name}
              </Badge>
            </a>
          ) : (
            <Badge variant="outline" className="text-[10px]">
              {model.license_name}
            </Badge>
          ))}
      </div>
    </div>
  );
}

function PresetCell({
  setup,
  currencySymbol,
}: {
  setup: UnrankedModelRow["tierSetups"][number];
  currencySymbol: string;
}) {
  if (!setup) {
    return (
      <div className="text-xs italic text-muted-foreground/50">Exceeds capacity</div>
    );
  }
  return (
    <div>
      <div className="text-sm font-medium">
        {setup.gpuCount}× {setup.gpuName}
        {setup.isProjected && " (*)"}
        {setup.gpuCount > 1 && isNvLink(setup.interconnect) && (
          <span className="text-muted-foreground font-normal ml-1.5">NVLink</span>
        )}
      </div>
      <div className="text-xs text-muted-foreground">
        {formatCurrency(setup.monthlyCost, currencySymbol)}/mo
      </div>
    </div>
  );
}

function ModelCard({
  row,
  currencySymbol,
}: {
  row: UnrankedModelRow;
  currencySymbol: string;
}) {
  return (
    <div className="rounded-lg border p-3">
      <ModelHeader model={row.model} />
      <hr className="my-3 border-border" />
      {/* GPU presets: cheapest setup per concurrency tier */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 tabular-nums">
        {CONCURRENCY_TIERS.map((tier, i) => (
          <div key={tier.key}>
            <div className="text-[11px] font-medium text-muted-foreground mb-1">
              {tier.label}
              <span className="font-normal"> · {tier.midpoint} streams</span>
            </div>
            <PresetCell setup={row.tierSetups[i]} currencySymbol={currencySymbol} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function UnrankedModelsSection({
  gpus,
  models,
  benchmarks,
  benchmarkCategory,
  settings,
  currencySymbol = "$",
}: UnrankedModelsSectionProps) {
  const rows = useMemo(
    () => calculateUnrankedModelRows(gpus, models, benchmarks, benchmarkCategory, settings),
    [gpus, models, benchmarks, benchmarkCategory, settings],
  );

  if (rows.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Unranked Models</CardTitle>
        <CardDescription>
          Open models with known size but no OpenHands Index result yet, so they can&apos;t
          be ranked above. Shown with the cheapest GPU setup per concurrency tier, plus
          license and HuggingFace links. The asterisk (*) marks projected multi-GPU setups
          beyond available offerings.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {rows.map((row) => (
            <ModelCard
              key={row.model.model_name}
              row={row}
              currencySymbol={currencySymbol}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
