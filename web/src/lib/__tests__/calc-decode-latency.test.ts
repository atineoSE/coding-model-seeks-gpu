import { describe, it, expect } from "vitest";
import {
  calcDecodeLatency,
  bandwidthRooflineTokS,
  type DecodeLatencyGpu,
  type DecodeLatencyModelDims,
} from "../calc-decode-latency";

// A representative MoE model (DeepSeek-style) sharded across multiple GPUs.
const MODEL: DecodeLatencyModelDims = {
  numLayers: 61,
  hiddenSize: 7168,
  activeParamsB: 37,
  bytesPerParam: 1, // fp8 serving
  isMoe: true,
  topK: 8,
  numExperts: 256,
  kvLoraRank: 512,
  qkRopeHeadDim: 64,
};

// Same HBM bandwidth on both ends so the only difference is the interconnect.
const HBM_TB_S = 3.35; // H100-class HBM3

const NVSWITCH: DecodeLatencyGpu = {
  hbmBandwidthTbS: HBM_TB_S,
  interconnectTier: "nvswitch",
  interconnectBandwidthGbS: 450, // NVLink 4 unidirectional
};

const NONE: DecodeLatencyGpu = {
  hbmBandwidthTbS: HBM_TB_S,
  interconnectTier: "none",
  interconnectBandwidthGbS: 64, // PCIe 5.0 x16 unidirectional
};

const TP = 4;

describe("calcDecodeLatency", () => {
  it("(a) gives lower single-stream tok/s on 'none' than on 'nvswitch'", () => {
    const none = calcDecodeLatency(NONE, MODEL, { tp: TP });
    const nvswitch = calcDecodeLatency(NVSWITCH, MODEL, { tp: TP });
    expect(none.singleStreamTokS).toBeLessThan(nvswitch.singleStreamTokS);
  });

  it("(b) lands strictly below the pure bandwidth roofline", () => {
    for (const gpu of [NONE, NVSWITCH]) {
      const result = calcDecodeLatency(gpu, MODEL, { tp: TP });
      const roofline = bandwidthRooflineTokS(gpu, MODEL, TP);
      expect(result.bandwidthRooflineTokS).toBeCloseTo(roofline, 6);
      expect(result.singleStreamTokS).toBeLessThan(roofline);
    }
  });

  it("stays strictly below the roofline even at tp=1 (launch overhead only)", () => {
    const result = calcDecodeLatency(NVSWITCH, MODEL, { tp: 1 });
    const roofline = bandwidthRooflineTokS(NVSWITCH, MODEL, 1);
    expect(result.breakdown.tpAllReduceS).toBe(0);
    expect(result.breakdown.epAllToAllS).toBe(0);
    expect(result.breakdown.launchS).toBeGreaterThan(0);
    expect(result.singleStreamTokS).toBeLessThan(roofline);
  });

  it("charges the EP all-to-all term only for MoE models", () => {
    const dense: DecodeLatencyModelDims = { ...MODEL, isMoe: false, topK: null };
    const moe = calcDecodeLatency(NVSWITCH, MODEL, { tp: TP });
    const denseResult = calcDecodeLatency(NVSWITCH, dense, { tp: TP });
    expect(moe.breakdown.epAllToAllS).toBeGreaterThan(0);
    expect(denseResult.breakdown.epAllToAllS).toBe(0);
  });

  it("breakdown terms sum to the total per-token time", () => {
    const { perTokenS, breakdown } = calcDecodeLatency(NONE, MODEL, {
      tp: TP,
      pp: 2,
      contextTokens: 51_000,
    });
    const sum =
      breakdown.weightReadS +
      breakdown.tpAllReduceS +
      breakdown.epAllToAllS +
      breakdown.launchS +
      breakdown.ppSendRecvS +
      breakdown.kvReadS;
    expect(sum).toBeCloseTo(perTokenS, 12);
    expect(perTokenS).toBeGreaterThan(0);
  });

  describe("KV-read term", () => {
    // GQA model with an explicit per-token KV footprint (bf16: 2·layers·kvH·hd·2).
    const GQA: DecodeLatencyModelDims = {
      ...MODEL,
      kvLoraRank: null,
      qkRopeHeadDim: null,
      numKvHeads: 8,
      kvBytesPerToken: 2 * 61 * 8 * 128 * 2,
    };

    it("is zero without context or KV width, positive with both", () => {
      expect(calcDecodeLatency(NVSWITCH, GQA, { tp: TP }).breakdown.kvReadS).toBe(0);
      expect(
        calcDecodeLatency(NVSWITCH, GQA, { tp: TP, contextTokens: 51_000 }).breakdown.kvReadS,
      ).toBeGreaterThan(0);
    });

    it("scales linearly with context length", () => {
      const a = calcDecodeLatency(NVSWITCH, GQA, { tp: TP, contextTokens: 25_000 });
      const b = calcDecodeLatency(NVSWITCH, GQA, { tp: TP, contextTokens: 50_000 });
      expect(b.breakdown.kvReadS).toBeCloseTo(2 * a.breakdown.kvReadS, 12);
    });

    it("charges the full context KV read, sharded across the KV-head TP group", () => {
      const tp = 4;
      const ctx = 51_000;
      const result = calcDecodeLatency(NVSWITCH, GQA, { tp, contextTokens: ctx });
      // kvShard = min(tp, numKvHeads) = min(4, 8) = 4; full read, no discount.
      const expected =
        (GQA.kvBytesPerToken! * ctx) / Math.min(tp, GQA.numKvHeads!) / (HBM_TB_S * 1024 ** 4);
      expect(result.breakdown.kvReadS).toBeCloseTo(expected, 12);
    });
  });
});
