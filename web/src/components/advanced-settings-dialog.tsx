"use client";

import { useState } from "react";
import { Settings } from "lucide-react";
import type { AdvancedSettings } from "@/types";
import { DEFAULT_ADVANCED_SETTINGS } from "@/lib/matrix-calculator";
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
import { Label } from "@/components/ui/label";

interface AdvancedSettingsDialogProps {
  settings: AdvancedSettings;
  onSettingsChange: (settings: AdvancedSettings) => void;
}

export function AdvancedSettingsDialog({
  settings,
  onSettingsChange,
}: AdvancedSettingsDialogProps) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<AdvancedSettings>(settings);

  function handleOpen(isOpen: boolean) {
    if (isOpen) {
      setDraft(settings);
    }
    setOpen(isOpen);
  }

  function handleSave() {
    onSettingsChange(draft);
    setOpen(false);
  }

  function handleReset() {
    setDraft(DEFAULT_ADVANCED_SETTINGS);
  }

  function updateNumericField(field: keyof AdvancedSettings, value: string) {
    const num = parseFloat(value);
    if (!isNaN(num) && num >= 0) {
      setDraft((prev) => ({ ...prev, [field]: num }));
    }
  }

  const fields: { key: keyof AdvancedSettings; label: string; description: string; step: string; min: string; max?: string }[] = [
    {
      key: "avgInputTokens",
      label: "Avg input tokens",
      description: "Average tokens per prompt (including conversation context)",
      step: "100",
      min: "0",
    },
    {
      key: "avgOutputTokens",
      label: "Avg output tokens",
      description: "Average tokens per model response",
      step: "100",
      min: "0",
    },
    {
      key: "minTokPerStream",
      label: "Min tok/s per stream",
      description: "Minimum acceptable decode throughput per concurrent stream",
      step: "5",
      min: "0",
    },
    {
      key: "prefixCacheHitRate",
      label: "Prefix cache hit rate (%)",
      description: "Percentage of each request already cached from previous requests",
      step: "5",
      min: "0",
      max: "90",
    },
  ];

  const inputClass = "flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs transition-colors placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";

  return (
    <Dialog open={open} onOpenChange={handleOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon-sm" title="Advanced Settings">
          <Settings className="size-4" />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Advanced Settings</DialogTitle>
          <DialogDescription>
            Adjust the parameters used for capacity and cost calculations.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-2">
          {fields.map((f) => (
            <div key={f.key} className="grid grid-cols-2 items-center gap-4">
              <div>
                <Label>{f.label}</Label>
                <p className="text-xs text-muted-foreground mt-0.5">{f.description}</p>
              </div>
              <input
                type="number"
                step={f.step}
                min={f.min}
                max={f.max}
                value={draft[f.key]}
                onChange={(e) => updateNumericField(f.key, e.target.value)}
                className={inputClass}
              />
            </div>
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={handleReset}>
            Reset to Defaults
          </Button>
          <Button onClick={handleSave}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
