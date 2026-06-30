"""Tests for interconnect classification in the gpuhunt source."""

from pipeline.sources.gpuhunt_source import classify_interconnect, resolve_gpu_name


def test_single_gpu_reports_board_fabric_never_none():
    """A single-GPU offering still reports its board fabric, never None."""
    # SXM-class card → nvlink; consumer card → pcie. The point is: not None.
    assert classify_interconnect("H100", "NVIDIA H100 80GB HBM3", 1) == "nvlink"
    assert classify_interconnect("RTX4090", "NVIDIA GeForce RTX 4090", 1) == "pcie"


def test_classify_interconnect_never_returns_none():
    """No input combination folds back to None."""
    for count in (1, 2, 8):
        for name, inst in [
            ("H100", "NVIDIA H100 PCIe"),
            ("H100", "NVIDIA H100 80GB HBM3"),
            ("RTX4090", "NVIDIA GeForce RTX 4090"),
            ("FutureGPU999", "some-instance"),
        ]:
            assert classify_interconnect(name, inst, count) in ("nvlink", "pcie")


def test_pcie_variant_remap():
    """An explicit PCIe instance remaps the SXM-class name to its variant key."""
    assert resolve_gpu_name("H100", "NVIDIA H100 PCIe") == "H100_PCIe"
    # SXM / unlabelled H100 stays on the base (SXM) key.
    assert resolve_gpu_name("H100", "NVIDIA H100 80GB HBM3") == "H100"
    assert resolve_gpu_name("H100", "8H100.80S.176V") == "H100"
    # Non-variant GPUs are never remapped.
    assert resolve_gpu_name("A100", "NVIDIA A100 PCIe") == "A100"


def test_explicit_pcie_wins_over_nvlink_class():
    """An explicit PCIe label beats the SXM-class assumption."""
    # Cheapest multi-GPU H100 on the market is the PCIe variant.
    assert classify_interconnect("H100", "NVIDIA H100 PCIe", 4) == "pcie"


def test_explicit_nvlink_signals():
    """SXM / HGX / DGX / NVL markers classify as nvlink."""
    assert classify_interconnect("A100", "NVIDIA A100-SXM4-80GB", 4) == "nvlink"
    assert classify_interconnect("H100NVL", "NVIDIA H100 NVL", 2) == "nvlink"
    assert classify_interconnect("GB200", "BM.GPU.GB200.4 (NVL72)", 4) == "nvlink"


def test_datacenter_class_assumed_nvlink_when_unlabelled():
    """Unlabelled datacenter cards fall back to their SXM form factor."""
    assert classify_interconnect("A100", "a2-megagpu-16g", 16) == "nvlink"
    assert classify_interconnect("B200", "p6-b200.48xlarge", 8) == "nvlink"
    assert classify_interconnect("H200", "8H200.141S.176V", 8) == "nvlink"


def test_consumer_cards_default_pcie():
    """Consumer / workstation cards default to PCIe."""
    assert classify_interconnect("RTX4090", "NVIDIA GeForce RTX 4090", 8) == "pcie"
    assert classify_interconnect("L40S", "g6e.12xlarge", 4) == "pcie"
    assert classify_interconnect("RTXPRO6000", "rtxpro6000mq-11-56-850-1lg.6", 4) == "pcie"


def test_unknown_card_defaults_pcie():
    """An unrecognised GPU conservatively defaults to PCIe (never over-estimates)."""
    assert classify_interconnect("FutureGPU999", "some-instance.4x", 4) == "pcie"
