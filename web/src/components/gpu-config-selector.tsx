"use client";

import { useState } from "react";
import type { PresetGpuConfig } from "@/types";
import { GPU_THROUGHPUT_SPECS, getGpuVram, gpuInterconnectTier } from "@/lib/gpu-specs";
import { interconnectBadgeLabel, formatInterconnectTier } from "@/components/deployment-estimate-panel";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

interface GpuConfigSelectorProps {
  value: PresetGpuConfig | null;
  onChange: (config: PresetGpuConfig) => void;
  presets: PresetGpuConfig[];
}

export function GpuConfigSelector({ value, onChange, presets }: GpuConfigSelectorProps) {
  const [customOpen, setCustomOpen] = useState(false);
  const [customGpu, setCustomGpu] = useState("H100");
  const [customCount, setCustomCount] = useState("4");

  const parsedCount = parseInt(customCount) || 1;

  const valueBadge = value ? interconnectBadgeLabel(value.gpuName, value.gpuCount) : null;

  function handleCustomSave() {
    const gpuName = customGpu;
    const count = parsedCount;
    const vram = getGpuVram(gpuName) ?? 80;

    onChange({
      label: `${count}× ${gpuName} ${vram}GB (Custom)`,
      gpuName,
      gpuCount: count,
      vramPerGpu: vram,
      totalVramGb: vram * count,
      // Interconnect tier is a property of the GPU (datasheet), not stored here.
      interconnect: null,
    });
    setCustomOpen(false);
  }

  const gpuTypes = Object.keys(GPU_THROUGHPUT_SPECS);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
        {presets.map((preset) => {
          const badge = interconnectBadgeLabel(preset.gpuName, preset.gpuCount);
          return (
          <button
            key={preset.label}
            onClick={() => onChange(preset)}
            className={cn(
              "rounded-lg border px-3 py-2.5 text-sm font-medium transition-all cursor-pointer",
              "hover:border-primary/50",
              value?.label === preset.label
                ? "border-primary bg-primary/5 shadow-sm"
                : "border-border bg-card",
            )}
          >
            {preset.label}
            {badge && (
              <Badge variant="secondary" className="text-[10px] ml-1.5">{badge}</Badge>
            )}
          </button>
          );
        })}
        <Dialog open={customOpen} onOpenChange={setCustomOpen}>
          <DialogTrigger asChild>
            <button
              className={cn(
                "rounded-lg border border-dashed px-3 py-2.5 text-sm font-medium transition-all cursor-pointer",
                "hover:border-primary/50 text-muted-foreground hover:text-foreground",
                value && !presets.find((p) => p.label === value.label)
                  ? "border-primary bg-primary/5"
                  : "border-border",
              )}
            >
              Custom...
            </button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Custom GPU Configuration</DialogTitle>
              <DialogDescription>
                Define your own GPU setup.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-2">
              <div className="grid grid-cols-2 items-center gap-4">
                <Label>GPU Type</Label>
                <Select value={customGpu} onValueChange={setCustomGpu}>
                  <SelectTrigger className="w-full min-w-0">
                    <SelectValue className="min-w-0" />
                  </SelectTrigger>
                  <SelectContent>
                    {gpuTypes.map((gpu) => (
                      <SelectItem key={gpu} value={gpu}>
                        {gpu} ({getGpuVram(gpu) ?? "?"}GB) · {formatInterconnectTier(gpuInterconnectTier(gpu))}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-2 items-center gap-4">
                <Label>Number of GPUs</Label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={customCount}
                  onChange={(e) => setCustomCount(e.target.value)}
                  className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setCustomOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleCustomSave}>Apply</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
      {value && (
        <p className="text-sm text-muted-foreground">
          Selected: <span className="font-medium text-foreground">{value.label}</span>
          {" "}({value.totalVramGb}GB total VRAM)
          {valueBadge && (
            <Badge variant="secondary" className="text-[10px] ml-1.5">{valueBadge}</Badge>
          )}
        </p>
      )}
    </div>
  );
}
