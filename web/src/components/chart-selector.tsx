"use client";

import { useState, type ReactNode } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

export interface ChartTab {
  value: string;
  label: string;
  content: ReactNode;
}

interface ChartSelectorProps {
  tabs: ChartTab[];
  defaultValue?: string;
}

export function ChartSelector({ tabs, defaultValue }: ChartSelectorProps) {
  const initial = defaultValue ?? tabs[0]?.value ?? "";

  return (
    <Tabs defaultValue={initial}>
      <div className="overflow-x-auto -mx-1 px-1">
        <TabsList variant="line" className="w-full justify-start">
          {tabs.map((tab) => (
            <TabsTrigger key={tab.value} value={tab.value} className="text-xs sm:text-sm">
              {tab.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>
      {tabs.map((tab) => (
        <TabsContent key={tab.value} value={tab.value}>
          {tab.content}
        </TabsContent>
      ))}
    </Tabs>
  );
}
