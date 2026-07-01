import { describe, it, expect } from "vitest";
import type { DeploymentEstimate } from "@/types";
import {
  formatTokS,
  formatStreamBand,
  formatInterconnectTier,
  formatContextAssumption,
  describeAssumptions,
  interconnectBadgeLabel,
} from "../deployment-estimate-panel";

describe("formatTokS", () => {
  it("rounds small values to whole tok/s", () => {
    expect(formatTokS(342.7)).toBe("343 tok/s");
    expect(formatTokS(0)).toBe("0 tok/s");
  });

  it("uses a k suffix with one decimal for thousands", () => {
    expect(formatTokS(1234)).toBe("1.2k tok/s");
    expect(formatTokS(1000)).toBe("1.0k tok/s");
  });

  it("drops the decimal for very large aggregates", () => {
    expect(formatTokS(123456)).toBe("123k tok/s");
  });
});

describe("formatStreamBand", () => {
  it("renders a low–high band with an en dash", () => {
    expect(formatStreamBand({ low: 4, high: 12 })).toBe("4–12");
  });

  it("collapses to a single number when low equals high", () => {
    expect(formatStreamBand({ low: 8, high: 8 })).toBe("8");
  });
});

describe("formatInterconnectTier", () => {
  it("maps each tier to a human label", () => {
    expect(formatInterconnectTier("nvswitch")).toBe("NVSwitch");
    expect(formatInterconnectTier("nvlink_paired")).toBe("NVLink+PCIe");
    expect(formatInterconnectTier("none")).toBe("PCIe");
  });
});

describe("interconnectBadgeLabel", () => {
  it("resolves a legacy string against the GPU datasheet tier", () => {
    // H100 datasheet tier is nvswitch; "nvlink" isn't a tier name → falls through.
    expect(interconnectBadgeLabel("nvlink", "H100")).toBe("NVSwitch");
    // H100_PCIe datasheet tier is nvlink_paired.
    expect(interconnectBadgeLabel(null, "H100_PCIe")).toBe("NVLink+PCIe");
  });

  it("honors an explicit tier override", () => {
    expect(interconnectBadgeLabel("nvlink_paired", "H100")).toBe("NVLink+PCIe");
    expect(interconnectBadgeLabel("nvswitch", "H100_PCIe")).toBe("NVSwitch");
  });

  it("returns null for PCIe-only setups (no badge)", () => {
    expect(interconnectBadgeLabel(null, "L40S")).toBeNull();
    expect(interconnectBadgeLabel("none", "H100")).toBeNull();
  });
});

describe("formatContextAssumption", () => {
  it("abbreviates token counts in K", () => {
    expect(
      formatContextAssumption({
        avgInputTokens: 50_000,
        avgOutputTokens: 1000,
      }),
    ).toBe("50K in / 1K out");
  });

  it("keeps sub-1000 token counts verbatim", () => {
    expect(
      formatContextAssumption({
        avgInputTokens: 500,
        avgOutputTokens: 200,
      }),
    ).toBe("500 in / 200 out");
  });
});

describe("describeAssumptions", () => {
  const base: DeploymentEstimate = {
    singleStreamTokS: 100,
    operatingStreams: { low: 2, high: 8 },
    aggregateTokS: 800,
    prefillComputeTokS: 90000,
    throughputState: "modeled",
    assumptions: {
      context: { avgInputTokens: 50_000, avgOutputTokens: 1000 },
      interconnectTier: "nvswitch",
      moe: true,
    },
  };

  it("surfaces context, interconnect tier, and MoE label", () => {
    expect(describeAssumptions(base)).toEqual([
      "50K in / 1K out",
      "NVSwitch interconnect",
      "MoE (active params)",
    ]);
  });

  it("labels dense models as Dense", () => {
    const dense: DeploymentEstimate = {
      ...base,
      assumptions: { ...base.assumptions, moe: false, interconnectTier: "none" },
    };
    expect(describeAssumptions(dense)).toEqual([
      "50K in / 1K out",
      "PCIe interconnect",
      "Dense",
    ]);
  });
});
