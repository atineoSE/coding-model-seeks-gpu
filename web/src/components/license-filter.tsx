"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface LicenseFilterProps {
  licenses: string[];
  value: string;
  onChange: (value: string) => void;
}

export function LicenseFilter({ licenses, value, onChange }: LicenseFilterProps) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium text-muted-foreground">
        License
      </label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full sm:w-[220px]">
          <SelectValue placeholder="Select License" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem key="All" value="All">All</SelectItem>
          {licenses.map((license) => (
            <SelectItem key={license} value={license}>
              {license}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
