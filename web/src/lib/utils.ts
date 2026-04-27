import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format an OpenHands canonical model name for human display.
 * "claude-opus-4-7" → "Claude Opus 4.7"
 * "GPT-5.4"         → "GPT-5.4"  (already well-formed, unchanged)
 * "Gemini-3.1-Pro"  → "Gemini 3.1 Pro"
 */
export function formatModelName(name: string): string {
  return name
    .replace(/(\d)-(\d)/g, "$1.$2") // digit-digit hyphens → dots (version numbers)
    .replace(/-/g, " ")
    .replace(/\b[a-z]/g, (c) => c.toUpperCase());
}
