"use client";

import { useState } from "react";
import type { PresetGpuConfig } from "@/types";
import { GPU_PRESETS } from "@/lib/gpu-presets";
import { GPU_THROUGHPUT_SPECS, getGpuThroughputSpec, getGpuVram } from "@/lib/gpu-specs";
import { isNvLink } from "@/lib/calculations";
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
}

export function GpuConfigSelector({ value, onChange }: GpuConfigSelectorProps) {
  const [customOpen, setCustomOpen] = useState(false);
  const [customGpu, setCustomGpu] = useState("H100");
  const [customCount, setCustomCount] = useState("4");
  const [customInterconnect, setCustomInterconnect] = useState<string>("nvlink");

  function handleGpuChange(gpu: string) {
    setCustomGpu(gpu);
    const spec = getGpuThroughputSpec(gpu);
    setCustomInterconnect(spec?.nvlink_bandwidth_gb_s ? "nvlink" : "pcie");
  }

  const parsedCount = parseInt(customCount) || 1;
  const showInterconnect = parsedCount > 1;

  function handleCustomSave() {
    const gpuName = customGpu;
    const count = parsedCount;
    const vram = getGpuVram(gpuName) ?? 80;

    const interconnect = count > 1 && customInterconnect === "nvlink" ? "nvlink" : null;

    onChange({
      label: `${count}× ${gpuName} ${vram}GB (Custom)`,
      gpuName,
      gpuCount: count,
      vramPerGpu: vram,
      totalVramGb: vram * count,
      interconnect,
    });
    setCustomOpen(false);
  }

  const gpuTypes = Object.keys(GPU_THROUGHPUT_SPECS);

  return (
    <div className="space-y-4">
      <label className="text-sm font-medium text-muted-foreground">
        Select your GPU configuration
      </label>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
        {GPU_PRESETS.map((preset) => (
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
            {isNvLink(preset.interconnect) && (
              <Badge variant="secondary" className="text-[10px] ml-1.5">NVLink</Badge>
            )}
          </button>
        ))}
        <Dialog open={customOpen} onOpenChange={setCustomOpen}>
          <DialogTrigger asChild>
            <button
              className={cn(
                "rounded-lg border border-dashed px-3 py-2.5 text-sm font-medium transition-all cursor-pointer",
                "hover:border-primary/50 text-muted-foreground hover:text-foreground",
                value && !GPU_PRESETS.find((p) => p.label === value.label)
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
                <Select value={customGpu} onValueChange={handleGpuChange}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {gpuTypes.map((gpu) => (
                      <SelectItem key={gpu} value={gpu}>
                        {gpu} ({getGpuVram(gpu) ?? "?"}GB)
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
              {showInterconnect && (
                <div className="grid grid-cols-2 items-center gap-4">
                  <Label>GPU Interconnect</Label>
                  <Select value={customInterconnect} onValueChange={setCustomInterconnect}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="pcie">PCIe</SelectItem>
                      <SelectItem value="nvlink">NVLink</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
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
          {isNvLink(value.interconnect) && (
            <Badge variant="secondary" className="text-[10px] ml-1.5">NVLink</Badge>
          )}
        </p>
      )}
    </div>
  );
}
