"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface BenchmarkCategory {
  name: string;
  displayName: string;
}

const CATEGORY_INFO: Record<string, { description: string; href: string; label: string }> = {
  overall:               { description: "Average of all 5 categories", href: "https://index.openhands.dev", label: "OpenHands Index" },
  issue_resolution:      { description: "SWE-bench Verified \u2014 500 instances", href: "https://www.swebench.com/", label: "SWE-bench Verified" },
  frontend:              { description: "SWE-bench Multimodal \u2014 617 instances", href: "https://github.com/OpenHands/SWE-bench-multimodal", label: "SWE-bench Multimodal" },
  greenfield:            { description: "Commit0 \u2014 16 libraries (lite split)", href: "https://github.com/commit-0/commit0", label: "Commit0" },
  testing:               { description: "SWT-bench Verified \u2014 433 instances", href: "https://github.com/logic-star-ai/swt-bench", label: "SWT-bench Verified" },
  information_gathering: { description: "GAIA \u2014 165 questions (validation split)", href: "https://huggingface.co/gaia-benchmark", label: "GAIA" },
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
          {info.description} &mdash;{" "}
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
