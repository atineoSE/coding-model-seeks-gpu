"""Tests for the shared node-reduction logic."""

from pipeline.gpu_nodes import SERVING_NODES, reduce_offerings_to_nodes


def _offering(gpu_name, gpu_count, price_per_hour, provider, vram_gb=80.0):
    return {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "gpu_count": gpu_count,
        "total_vram_gb": vram_gb * gpu_count,
        "price_per_hour": price_per_hour,
        "currency": "USD",
        "provider": provider,
        "instance_name": f"{gpu_name}_{gpu_count}x",
        "location": "us-east-1",
        "interconnect": "nvlink",
    }


def test_serving_nodes_constant():
    """The curated order is fixed."""
    assert SERVING_NODES == ["B300", "B200", "H200", "H100", "A100", "RTXPRO6000"]


def test_cheapest_wins_per_node():
    """The min price_per_hour offering is chosen per node."""
    offerings = [
        _offering("H100", 8, 25.0, "expensive"),
        _offering("H100", 8, 18.32, "verda"),
        _offering("H100", 8, 20.0, "midrange"),
    ]
    result = reduce_offerings_to_nodes(offerings)
    assert result == [
        {"gpu_name": "H100", "usd_per_node_hour": 18.32},
    ]


def test_a100_40gb_excluded_80gb_kept():
    """A100 uses only the 80GB variant even if a cheaper 40GB row exists."""
    offerings = [
        _offering("A100", 8, 10.0, "cheap40", vram_gb=40.0),
        _offering("A100", 8, 15.0, "kept80", vram_gb=80.0),
    ]
    result = reduce_offerings_to_nodes(offerings)
    assert result == [
        {"gpu_name": "A100", "usd_per_node_hour": 15.0},
    ]


def test_non_8x_rows_ignored():
    """Only gpu_count == 8 offerings become a node."""
    offerings = [
        _offering("H200", 1, 3.0, "single"),
        _offering("H200", 2, 6.0, "double"),
        _offering("H200", 4, 12.0, "quad"),
        _offering("H200", 8, 22.0, "node"),
    ]
    result = reduce_offerings_to_nodes(offerings)
    assert result == [
        {"gpu_name": "H200", "usd_per_node_hour": 22.0},
    ]


def test_node_with_no_8x_offering_omitted():
    """A curated GPU present only in non-8x rows is dropped entirely."""
    offerings = [
        _offering("B300", 4, 40.0, "b300quad"),  # no 8x -> omitted
        _offering("B200", 8, 35.0, "b200node"),
    ]
    result = reduce_offerings_to_nodes(offerings)
    assert result == [
        {"gpu_name": "B200", "usd_per_node_hour": 35.0},
    ]


def test_output_ordering_follows_serving_nodes():
    """Output is sorted by SERVING_NODES order regardless of input order."""
    offerings = [
        _offering("RTXPRO6000", 8, 8.0, "rtx"),
        _offering("H100", 8, 18.0, "h100"),
        _offering("B300", 8, 50.0, "b300"),
        _offering("A100", 8, 12.0, "a100"),
    ]
    result = reduce_offerings_to_nodes(offerings)
    assert [row["gpu_name"] for row in result] == ["B300", "H100", "A100", "RTXPRO6000"]


def test_robust_to_missing_optional_keys():
    """Older-schema dicts missing non-essential keys still reduce cleanly."""
    offerings = [
        {
            "gpu_name": "H100",
            "gpu_count": 8,
            "vram_gb": 80.0,
            "price_per_hour": 19.5,
            "provider": "legacy",
        },
    ]
    result = reduce_offerings_to_nodes(offerings)
    assert result == [
        {"gpu_name": "H100", "usd_per_node_hour": 19.5},
    ]


def test_non_curated_gpu_ignored():
    """Offerings for GPUs outside the curated set are skipped."""
    offerings = [
        _offering("A10", 8, 5.0, "a10"),
        _offering("H100", 8, 18.0, "h100"),
    ]
    result = reduce_offerings_to_nodes(offerings)
    assert result == [
        {"gpu_name": "H100", "usd_per_node_hour": 18.0},
    ]
