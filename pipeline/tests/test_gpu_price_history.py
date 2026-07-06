"""Tests for the daily GPU node price-history exporter."""

import json
from datetime import UTC, datetime

from pipeline.exporters import json_export
from pipeline.exporters.json_export import export_gpu_price_history


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


def _today():
    return datetime.now(UTC).date().isoformat()


def test_creates_fresh_file(tmp_path):
    """A missing history file is created with the expected skeleton + today's row."""
    offerings = [
        _offering("H100", 8, 18.32, "verda"),
        _offering("B200", 8, 35.0, "b200node"),
    ]
    path = export_gpu_price_history(offerings, output_dir=tmp_path)

    assert path == tmp_path / "gpu_price_history.json"
    data = json.loads(path.read_text())
    assert data["unit"] == "usd_per_node_hour"
    assert "generated_at" in data
    assert len(data["series"]) == 1

    entry = data["series"][0]
    assert entry["date"] == _today()
    assert entry["prices"] == [
        {"gpu_name": "B200", "usd_per_node_hour": 35.0},
        {"gpu_name": "H100", "usd_per_node_hour": 18.32},
    ]


def test_appends_snapshot_when_prices_change(tmp_path):
    """A pre-existing snapshot is preserved and today's row is appended when prices move."""
    path = tmp_path / "gpu_price_history.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "unit": "usd_per_node_hour",
                "series": [
                    {
                        "date": "2020-01-01",
                        "prices": [{"gpu_name": "H100", "usd_per_node_hour": 99.0}],
                    }
                ],
            }
        )
    )

    offerings = [_offering("H100", 8, 18.0, "verda")]
    export_gpu_price_history(offerings, output_dir=tmp_path)

    data = json.loads(path.read_text())
    dates = [e["date"] for e in data["series"]]
    assert dates == ["2020-01-01", _today()]
    # The old snapshot is untouched; today's snapshot reflects the new price.
    assert data["series"][0]["prices"] == [
        {"gpu_name": "H100", "usd_per_node_hour": 99.0}
    ]
    assert data["series"][1]["prices"] == [
        {"gpu_name": "H100", "usd_per_node_hour": 18.0},
    ]


def test_no_snapshot_when_prices_unchanged(tmp_path):
    """When today's prices match the latest snapshot, no new entry is added."""
    path = tmp_path / "gpu_price_history.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "unit": "usd_per_node_hour",
                "series": [
                    {
                        "date": "2020-01-01",
                        "prices": [{"gpu_name": "H100", "usd_per_node_hour": 18.0}],
                    }
                ],
            }
        )
    )
    first_bytes = path.read_bytes()

    # Same cheapest price as the latest snapshot -> nothing to record.
    offerings = [_offering("H100", 8, 18.0, "verda")]
    export_gpu_price_history(offerings, output_dir=tmp_path)

    data = json.loads(path.read_text())
    assert [e["date"] for e in data["series"]] == ["2020-01-01"]
    # Content is byte-identical: the unchanged-series guard skipped the write.
    assert path.read_bytes() == first_bytes


def test_idempotent_same_day_rerun_no_churn(tmp_path):
    """Re-running the same day with identical prices produces no duplicate and no write."""
    offerings = [_offering("H100", 8, 18.0, "verda")]
    path = export_gpu_price_history(offerings, output_dir=tmp_path)

    first_bytes = path.read_bytes()
    first_mtime = path.stat().st_mtime_ns

    export_gpu_price_history(offerings, output_dir=tmp_path)

    data = json.loads(path.read_text())
    today_rows = [e for e in data["series"] if e["date"] == _today()]
    assert len(today_rows) == 1  # no duplicate

    # Byte-identical and untouched on disk (write was skipped).
    assert path.read_bytes() == first_bytes
    assert path.stat().st_mtime_ns == first_mtime


def test_delegates_node_selection_to_reduce_offerings_to_nodes(tmp_path, monkeypatch):
    """The exporter obtains today's prices from reduce_offerings_to_nodes."""
    offerings = [_offering("H100", 8, 18.0, "verda")]
    sentinel_prices = [
        {"gpu_name": "SENTINEL", "usd_per_node_hour": 1.23},
    ]
    calls = []

    def fake_reduce(arg):
        calls.append(arg)
        return sentinel_prices

    monkeypatch.setattr(json_export, "reduce_offerings_to_nodes", fake_reduce)

    path = export_gpu_price_history(offerings, output_dir=tmp_path)

    assert calls == [offerings]
    data = json.loads(path.read_text())
    assert data["series"][0]["prices"] == sentinel_prices
