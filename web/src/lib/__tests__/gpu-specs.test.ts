import { describe, it, expect } from "vitest";
import { GPU_THROUGHPUT_SPECS, getGpuThroughputSpec } from "../gpu-specs";
import gpuSpecsData from "../../../public/data/gpu_specs.json";

const VALID_TIERS = new Set(["none", "nvlink_paired", "nvswitch"]);

describe("gpu_specs.json data shape", () => {
  it("parses and every entry carries the new physical-input fields", () => {
    expect(gpuSpecsData.length).toBeGreaterThan(0);
    for (const e of gpuSpecsData) {
      expect("pcie_bandwidth_gb_s" in e).toBe(true);
      expect("interconnect_tier" in e).toBe(true);
      expect("memory_type" in e).toBe(true);
      expect(VALID_TIERS.has(e.interconnect_tier)).toBe(true);
    }
  });

  it("derives interconnect_tier consistently with the NVLink signal", () => {
    for (const e of gpuSpecsData) {
      if (e.nvlink_bandwidth_gb_s == null) {
        expect(e.interconnect_tier).toBe("none");
      } else {
        expect(e.interconnect_tier).not.toBe("none");
      }
    }
  });

  it("populates a known datasheet fact (H100: HBM3, PCIe 5.0, NVSwitch)", () => {
    const h100 = gpuSpecsData.find((e) => e.gpu_name === "H100");
    expect(h100).toBeDefined();
    expect(h100!.memory_type).toBe("HBM3");
    expect(h100!.pcie_bandwidth_gb_s).toBe(64);
    expect(h100!.interconnect_tier).toBe("nvswitch");
  });
});

describe("getGpuThroughputSpec", () => {
  it("exposes the new fields on the loaded spec", () => {
    const spec = getGpuThroughputSpec("A100");
    expect(spec).not.toBeNull();
    expect(spec!.pcie_bandwidth_gb_s).toBe(32);
    expect(spec!.memory_type).toBe("HBM2e");
    expect(spec!.interconnect_tier).toBe("nvswitch");
  });

  it("classifies a PCIe-only GPU as interconnect_tier 'none'", () => {
    const spec = getGpuThroughputSpec("L40S");
    expect(spec).not.toBeNull();
    expect(spec!.nvlink_bandwidth_gb_s).toBeNull();
    expect(spec!.interconnect_tier).toBe("none");
  });

  it("classifies a bridged-pair workstation GPU as 'nvlink_paired'", () => {
    const spec = getGpuThroughputSpec("A6000");
    expect(spec).not.toBeNull();
    expect(spec!.interconnect_tier).toBe("nvlink_paired");
  });

  it("every loaded spec has a valid interconnect_tier", () => {
    for (const name of Object.keys(GPU_THROUGHPUT_SPECS)) {
      expect(VALID_TIERS.has(GPU_THROUGHPUT_SPECS[name].interconnect_tier)).toBe(true);
    }
  });
});
