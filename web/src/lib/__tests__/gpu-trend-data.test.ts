import { describe, it, expect } from "vitest";
import type { GpuNodePriceHistory } from "@/types";
import { computeGpuNodePriceTrend } from "../gpu-trend-data";

const HOURS_PER_MONTH = 720;

const FIXTURE: GpuNodePriceHistory = {
  generated_at: "2026-07-01T00:00:00Z",
  unit: "usd_per_node_hour",
  series: [
    {
      date: "2026-06-01",
      prices: [
        { gpu_name: "H100", usd_per_node_hour: 2 },
        { gpu_name: "A100", usd_per_node_hour: 1.5 },
      ],
    },
    {
      date: "2026-06-02",
      prices: [
        // H100 only this day — A100 is absent, so its key must be missing.
        { gpu_name: "H100", usd_per_node_hour: 1.8 },
      ],
    },
  ],
};

describe("computeGpuNodePriceTrend", () => {
  it("pivots the series into one row per date keyed by gpu name", () => {
    const rows = computeGpuNodePriceTrend(FIXTURE);
    expect(rows).toHaveLength(2);
    expect(rows[0].date).toBe("2026-06-01");
    expect(rows[1].date).toBe("2026-06-02");
  });

  it("converts stored hourly price to $/month (hourly × 720)", () => {
    const rows = computeGpuNodePriceTrend(FIXTURE);
    expect(rows[0].H100).toBe(2 * HOURS_PER_MONTH);
    expect(rows[0].A100).toBe(1.5 * HOURS_PER_MONTH);
    expect(rows[1].H100).toBe(1.8 * HOURS_PER_MONTH);
  });

  it("leaves a node's key absent on days it has no price", () => {
    const rows = computeGpuNodePriceTrend(FIXTURE);
    expect("A100" in rows[1]).toBe(false);
    expect(rows[1].A100).toBeUndefined();
  });

  it("returns an empty array when the series is empty", () => {
    const rows = computeGpuNodePriceTrend({
      generated_at: "2026-07-01T00:00:00Z",
      unit: "usd_per_node_hour",
      series: [],
    });
    expect(rows).toEqual([]);
  });
});
