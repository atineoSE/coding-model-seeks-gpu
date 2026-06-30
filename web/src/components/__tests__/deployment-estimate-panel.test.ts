import { describe, it, expect } from "vitest";
import type { DeploymentEstimate } from "@/types";
import {
  formatTokS,
  formatStreamBand,
  formatInterconnectTier,
  formatContextAssumption,
  describeAssumptions,
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
    expect(formatInterconnectTier("nvlink_paired")).toBe("NVLink");
    expect(formatInterconnectTier("none")).toBe("PCIe");
  });
});

describe("formatContextAssumption", () => {
  it("abbreviates token counts in K and shows prefix reuse as a percent", () => {
    expect(
      formatContextAssumption({
        avgInputTokens: 50_000,
        avgOutputTokens: 1000,
        prefixReuse: 0.5,
      }),
    ).toBe("50K in / 1K out · 50% prefix reuse");
  });

  it("keeps sub-1000 token counts verbatim", () => {
    expect(
      formatContextAssumption({
        avgInputTokens: 500,
        avgOutputTokens: 200,
        prefixReuse: 0.25,
      }),
    ).toBe("500 in / 200 out · 25% prefix reuse");
  });
});

describe("describeAssumptions", () => {
  const base: DeploymentEstimate = {
    singleStreamTokS: 100,
    operatingStreams: { low: 2, high: 8 },
    aggregateTokS: 800,
    throughputState: "modeled",
    assumptions: {
      context: { avgInputTokens: 50_000, avgOutputTokens: 1000, prefixReuse: 0.5 },
      interconnectTier: "nvswitch",
      moe: true,
    },
  };

  it("surfaces context, interconnect tier, and MoE label", () => {
    expect(describeAssumptions(base)).toEqual([
      "50K in / 1K out · 50% prefix reuse",
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
      "50K in / 1K out · 50% prefix reuse",
      "PCIe interconnect",
      "Dense",
    ]);
  });
});
