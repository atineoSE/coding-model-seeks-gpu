import { useState, useEffect } from "react";
import type { GpuNodePriceHistory } from "@/types";
import { monthlyCost } from "./calculations";

// ---------------------------------------------------------------------------
// GPU node price history — loading
// ---------------------------------------------------------------------------

/**
 * Fetch the stored GPU node price history from /data/gpu_price_history.json.
 * Mirrors the fetch/loading pattern in trend-data.ts (useSnapshotData).
 */
export function useGpuPriceHistory(): {
  history: GpuNodePriceHistory | null;
  loading: boolean;
} {
  const [history, setHistory] = useState<GpuNodePriceHistory | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data: GpuNodePriceHistory = await fetch(
          "/data/gpu_price_history.json",
        ).then((r) => r.json());
        if (!cancelled) {
          setHistory(data);
        }
      } catch {
        if (!cancelled) {
          setHistory(null);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return { history, loading };
}

// ---------------------------------------------------------------------------
// GPU node price trend — pivot into Recharts rows
// ---------------------------------------------------------------------------

/**
 * A pivoted Recharts row: a `date` plus one numeric key per GPU node holding
 * the $/month value for that day. A node's key is absent on days it has no
 * price (so Recharts draws a gap rather than a zero).
 */
export interface GpuNodePriceTrendRow {
  date: string;
  [gpuName: string]: number | string;
}

/**
 * Pivot the price-history series into Recharts rows: one row per snapshot date,
 * one numeric key per GPU node holding the $/month value (monthlyCost of the
 * stored hourly node price). Nodes without a price on a given day are simply
 * left out of that row.
 */
export function computeGpuNodePriceTrend(
  history: GpuNodePriceHistory,
): GpuNodePriceTrendRow[] {
  return history.series.map(({ date, prices }) => {
    const row: GpuNodePriceTrendRow = { date };
    for (const p of prices) {
      row[p.gpu_name] = monthlyCost(p.usd_per_node_hour);
    }
    return row;
  });
}
