import type { ConcurrencyTier, ConcurrencyTierConfig } from "@/types";

export const CONCURRENCY_TIERS: ConcurrencyTierConfig[] = [
  {
    key: "single_agent",
    label: "Single Agent",
    minStreams: 5,
    maxStreams: 5,
    midpoint: 5,
    description: "5 concurrent inference streams. A single developer or one autonomous agent.",
  },
  {
    key: "multi_agent",
    label: "Multiple Agents",
    minStreams: 20,
    maxStreams: 20,
    midpoint: 20,
    description: "20 concurrent streams. A small team or several agents in parallel.",
  },
  {
    key: "agent_fleet",
    label: "Agent Fleet",
    minStreams: 80,
    maxStreams: 80,
    midpoint: 80,
    description: "80 concurrent streams. A fleet of autonomous coding agents or a medium team.",
  },
  {
    key: "agent_swarm",
    label: "Agent Swarm",
    minStreams: 150,
    maxStreams: 150,
    midpoint: 150,
    description: "150 concurrent streams. Large-scale agent orchestration or production serving.",
  },
];

export function getConcurrencyTier(key: ConcurrencyTier): ConcurrencyTierConfig {
  return CONCURRENCY_TIERS.find((t) => t.key === key)!;
}

export function getConcurrencyTierLabel(key: ConcurrencyTier): string {
  return getConcurrencyTier(key).label;
}

export function formatTierRange(tier: ConcurrencyTierConfig): string {
  return `${tier.minStreams}\u2013${tier.maxStreams}`;
}
