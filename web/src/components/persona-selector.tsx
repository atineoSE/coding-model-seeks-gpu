"use client";

import type { Persona } from "@/types";
import { cn } from "@/lib/utils";

interface PersonaSelectorProps {
  value: Persona;
  onChange: (persona: Persona) => void;
}

const personas: { key: Persona; title: string; subtitle: string; icon: string }[] = [
  {
    key: "performance",
    title: "I want the best model",
    subtitle: "Find the top-performing coding LLM and the GPU setup you need",
    icon: "trophy",
  },
  {
    key: "budget",
    title: "I\u2019ve got GPUs",
    subtitle: "See what you can run on your existing GPU infrastructure",
    icon: "server",
  },
  {
    key: "trends",
    title: "I\u2019m watching the space",
    subtitle: "Draw insights from gaps in cost and performance",
    icon: "chart",
  },
];

export function PersonaSelector({ value, onChange }: PersonaSelectorProps) {
  return (
    <div className="grid grid-cols-3 gap-2 sm:gap-4 max-w-4xl mx-auto">
      {personas.map((p) => (
        <button
          key={p.key}
          onClick={() => onChange(p.key)}
          className={cn(
            "flex flex-col items-center gap-1 sm:gap-3 rounded-xl border-2 p-2 sm:p-6 text-center transition-all cursor-pointer",
            "hover:border-primary/50 hover:shadow-md",
            value === p.key
              ? "border-primary bg-primary/5 shadow-sm"
              : "border-border bg-card",
          )}
        >
          <div className={cn(
            "flex h-10 w-10 sm:h-12 sm:w-12 items-center justify-center rounded-full shrink-0",
            value === p.key ? "bg-primary/10" : "bg-muted",
          )}>
            {p.icon === "trophy" ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>
            ) : p.icon === "chart" ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/></svg>
            )}
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-xs sm:text-lg leading-tight">{p.title}</h3>
            <p className="text-sm text-muted-foreground mt-1 hidden sm:block">{p.subtitle}</p>
          </div>
        </button>
      ))}
    </div>
  );
}
