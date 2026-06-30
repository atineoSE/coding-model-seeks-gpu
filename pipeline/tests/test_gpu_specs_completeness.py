"""Every emitted GPU spec must carry a complete, valid interconnect tier and
memory metadata — the frontend topology model depends on it, so we never ship a
spec with a null/absent tier."""

from pipeline.sources.dbgpu_source import (
    _VALID_TIERS,
    GPUHUNT_TO_DBGPU_KEY,
    INTERCONNECT_TIER,
    fetch_gpu_specs,
)

_REQUIRED_FIELDS = (
    "interconnect_tier",
    "memory_type",
    "pcie_bandwidth_gb_s",
    "memory_bandwidth_tb_s",
)


def test_every_mapped_gpu_has_a_tier():
    """INTERCONNECT_TIER must cover every GPU in GPUHUNT_TO_DBGPU_KEY."""
    missing = set(GPUHUNT_TO_DBGPU_KEY) - set(INTERCONNECT_TIER)
    assert not missing, f"GPUs without an interconnect_tier: {sorted(missing)}"
    assert set(INTERCONNECT_TIER.values()) <= _VALID_TIERS


def test_fetched_specs_are_complete():
    """fetch_gpu_specs emits a valid, non-null value for every required field."""
    for spec in fetch_gpu_specs():
        name = spec["gpu_name"]
        for field in _REQUIRED_FIELDS:
            assert spec.get(field) is not None, f"{name} missing {field}"
        assert spec["interconnect_tier"] in _VALID_TIERS, name


def test_h100_sxm_and_pcie_are_distinct_skus():
    """H100 (SXM) and H100_PCIe must resolve to different bandwidth + tier."""
    specs = {s["gpu_name"]: s for s in fetch_gpu_specs()}
    assert specs["H100"]["memory_bandwidth_tb_s"] == 3.36
    assert specs["H100"]["interconnect_tier"] == "nvswitch"
    assert specs["H100_PCIe"]["memory_bandwidth_tb_s"] == 2.04
    assert specs["H100_PCIe"]["memory_type"] == "HBM2e"
    assert specs["H100_PCIe"]["interconnect_tier"] == "nvlink_paired"
