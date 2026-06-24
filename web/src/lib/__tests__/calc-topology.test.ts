import { describe, it, expect } from "vitest";
import { calcTopology, NVLINK_PAIR_SIZE } from "../calc-topology";

// A model that comfortably fits across the layouts under test.
const FITS = { modelSizeGb: 160, vramPerGpuGb: 80 }; // 8 GPUs → 640 GB available

describe("calcTopology — nvswitch tier", () => {
  it("spans tensor parallelism across all GPUs with no forced pipeline", () => {
    const layout = calcTopology({
      interconnectTier: "nvswitch",
      gpuCount: 8,
      ...FITS,
    });
    expect(layout.tp).toBe(8);
    expect(layout.pp).toBe(1);
    expect(layout.gpusUsed).toBe(8);
    expect(layout.tpCrossesSlowLink).toBe(false);
    expect(layout.feasible).toBe(true);
  });

  it("allows any TP degree (odd GPU counts) on the all-to-all fabric", () => {
    const layout = calcTopology({
      interconnectTier: "nvswitch",
      gpuCount: 3,
      ...FITS,
    });
    expect(layout.tp).toBe(3);
    expect(layout.pp).toBe(1);
  });
});

describe("calcTopology — nvlink_paired tier", () => {
  it("caps TP at the pair size and forces PP across the pairs", () => {
    const layout = calcTopology({
      interconnectTier: "nvlink_paired",
      gpuCount: 8,
      ...FITS,
    });
    expect(layout.tp).toBe(NVLINK_PAIR_SIZE);
    expect(layout.tp).toBe(2);
    expect(layout.pp).toBe(4);
    expect(layout.gpusUsed).toBe(8);
    expect(layout.tpCrossesSlowLink).toBe(false);
    expect(layout.feasible).toBe(true);
  });

  it("degrades a single GPU to TP=1, PP=1", () => {
    const layout = calcTopology({
      interconnectTier: "nvlink_paired",
      gpuCount: 1,
      ...FITS,
    });
    expect(layout.tp).toBe(1);
    expect(layout.pp).toBe(1);
  });

  it("drops an odd leftover GPU rather than forming a broken pair", () => {
    const layout = calcTopology({
      interconnectTier: "nvlink_paired",
      gpuCount: 5,
      ...FITS,
    });
    expect(layout.tp).toBe(2);
    expect(layout.pp).toBe(2); // floor(5 / 2)
    expect(layout.gpusUsed).toBe(4);
  });
});

describe("calcTopology — none tier", () => {
  it("allows TP but flags the slow PCIe link for the latency penalty", () => {
    const layout = calcTopology({
      interconnectTier: "none",
      gpuCount: 4,
      ...FITS,
    });
    expect(layout.tp).toBe(4);
    expect(layout.pp).toBe(1);
    expect(layout.tpCrossesSlowLink).toBe(true);
    expect(layout.feasible).toBe(true);
  });

  it("does not flag a single-GPU layout (no all-reduce to cross PCIe)", () => {
    const layout = calcTopology({
      interconnectTier: "none",
      gpuCount: 1,
      ...FITS,
    });
    expect(layout.tp).toBe(1);
    expect(layout.tpCrossesSlowLink).toBe(false);
  });
});

describe("calcTopology — feasibility", () => {
  it("reports infeasible when weights exceed aggregate VRAM", () => {
    const layout = calcTopology({
      interconnectTier: "nvswitch",
      gpuCount: 2,
      modelSizeGb: 200,
      vramPerGpuGb: 80, // 160 GB total < 200 GB weights
    });
    expect(layout.feasible).toBe(false);
  });

  it("reports feasible at the exact VRAM boundary", () => {
    const layout = calcTopology({
      interconnectTier: "nvswitch",
      gpuCount: 2,
      modelSizeGb: 160,
      vramPerGpuGb: 80,
    });
    expect(layout.feasible).toBe(true);
  });
});

describe("calcTopology — input validation", () => {
  it("rejects a non-positive GPU count", () => {
    expect(() =>
      calcTopology({ interconnectTier: "nvswitch", gpuCount: 0, ...FITS }),
    ).toThrow();
  });

  it("rejects a non-positive per-GPU VRAM", () => {
    expect(() =>
      calcTopology({
        interconnectTier: "nvswitch",
        gpuCount: 2,
        modelSizeGb: 10,
        vramPerGpuGb: 0,
      }),
    ).toThrow();
  });
});
