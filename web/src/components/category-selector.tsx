"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { BENCHMARK_INSTANCE_COUNTS } from "@/lib/benchmark-costs";

export interface BenchmarkCategory {
  name: string;
  displayName: string;
}

const CATEGORY_INFO: Record<string, { description: string; href: string; label: string; unit?: string }> = {
  overall:               { description: "Average of all 5 categories", href: "https://index.openhands.dev", label: "OpenHands Index" },
  issue_resolution:      { description: "SWE-bench Verified", href: "https://www.swebench.com/", label: "SWE-bench Verified", unit: "instances" },
  frontend:              { description: "SWE-bench Multimodal", href: "https://www.swebench.com/multimodal.html", label: "SWE-bench Multimodal", unit: "instances" },
  greenfield:            { description: "Commit0", href: "https://github.com/commit-0/commit0", label: "Commit0", unit: "libraries (lite split)" },
  testing:               { description: "SWT-bench Verified", href: "https://github.com/logic-star-ai/swt-bench", label: "SWT-bench Verified", unit: "instances" },
  information_gathering: { description: "GAIA", href: "https://huggingface.co/gaia-benchmark", label: "GAIA", unit: "questions (validation split)" },
};

interface CategorySelectorProps {
  categories: BenchmarkCategory[];
  value: string;
  onChange: (value: string) => void;
}

export function CategorySelector({ categories, value, onChange }: CategorySelectorProps) {
  const info = CATEGORY_INFO[value];

  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-muted-foreground">
        Benchmark Category
      </label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full sm:w-[280px]">
          <SelectValue placeholder="Select Category" />
        </SelectTrigger>
        <SelectContent>
          {categories.map((cat) => (
            <SelectItem key={cat.name} value={cat.name}>
              {cat.displayName}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {info && (
        <p className="text-xs text-muted-foreground">
          {info.description}
          {BENCHMARK_INSTANCE_COUNTS[value] !== undefined && info.unit && (
            <> &mdash; {BENCHMARK_INSTANCE_COUNTS[value]} {info.unit}</>
          )}
          {" "}&mdash;{" "}
          <a
            href={info.href}
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-foreground transition-colors"
          >
            {info.label}
          </a>
        </p>
      )}
    </div>
  );
}
