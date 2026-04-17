import { describe, it, expect } from "vitest";
import { computeLabelOffsets } from "../model-size-chart";

describe("computeLabelOffsets", () => {
  it("returns zero offsets when labels are far apart", () => {
    const positions = [
      { x: 0, y: 0, key: "A" },
      { x: 200, y: 200, key: "B" },
    ];
    const offsets = computeLabelOffsets(positions);
    expect(offsets.get("A")).toBe(0);
    expect(offsets.get("B")).toBe(0);
  });

  it("nudges overlapping labels apart", () => {
    const positions = [
      { x: 100, y: 100, key: "A" },
      { x: 105, y: 105, key: "B" }, // within minDx=40 and minDy=12
    ];
    const offsets = computeLabelOffsets(positions);
    expect(offsets.get("A")).toBe(0);
    // B should be nudged further up (negative offset)
    expect(offsets.get("B")!).toBeLessThan(0);
  });

  it("does not nudge labels that are close in x but far in y", () => {
    const positions = [
      { x: 100, y: 0, key: "A" },
      { x: 110, y: 50, key: "B" }, // same x neighborhood, but dy=50 > 12
    ];
    const offsets = computeLabelOffsets(positions);
    expect(offsets.get("A")).toBe(0);
    expect(offsets.get("B")).toBe(0);
  });

  it("handles empty input", () => {
    const offsets = computeLabelOffsets([]);
    expect(offsets.size).toBe(0);
  });

  it("handles three overlapping labels", () => {
    const positions = [
      { x: 100, y: 100, key: "A" },
      { x: 102, y: 103, key: "B" },
      { x: 104, y: 106, key: "C" },
    ];
    const offsets = computeLabelOffsets(positions);
    expect(offsets.get("A")).toBe(0);
    // B and C should both be nudged, with C nudged more
    expect(offsets.get("B")!).toBeLessThan(0);
    expect(offsets.get("C")!).toBeLessThan(offsets.get("B")!);
  });
});
