#!/usr/bin/env python3
"""Backfill ``gpu_price_history.json`` from the git history of ``gpus.json``.

Walks the committed history of ``web/public/data/gpus.json``, takes the last
commit per UTC calendar day, reduces that day's offerings to curated node
prices via :func:`pipeline.gpu_nodes.reduce_offerings_to_nodes`, and writes the
assembled daily series to ``web/public/data/gpu_price_history.json``.

The script is idempotent: re-running fully regenerates the output file from the
current git history.

Run from the ``pipeline`` directory::

    cd pipeline && uv run python scripts/backfill_gpu_price_history.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_DIR.parent

# Make ``pipeline.gpu_nodes`` importable regardless of the invocation cwd.
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline.gpu_nodes import reduce_offerings_to_nodes  # noqa: E402

GPUS_JSON = "web/public/data/gpus.json"
OUTPUT_PATH = REPO_ROOT / "web/public/data/gpu_price_history.json"
UNIT = "usd_per_node_hour"


def _git(*args: str) -> str:
    """Run a git command at the repo root and return stdout (text)."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def last_commit_per_day() -> list[tuple[str, str]]:
    """Return ``(date, commit_hash)`` pairs, one per UTC calendar day.

    ``git log`` yields commits newest-first, so the first commit seen for a
    given UTC day is that day's last (most recent) commit.
    """
    log = _git("log", "--format=%H %cI", "--", GPUS_JSON)

    latest: dict[str, str] = {}
    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        commit_hash, _, iso = line.partition(" ")
        committed = datetime.fromisoformat(iso).astimezone(UTC)
        day = committed.date().isoformat()
        if day not in latest:
            latest[day] = commit_hash
    return sorted(latest.items())


def prices_for_commit(commit_hash: str) -> list[dict] | None:
    """Reduce a commit's ``gpus.json`` to node prices, or ``None`` to skip.

    Skips commits whose file is missing, fails to parse, is not a list, or
    predates the schema (yields no serving nodes).
    """
    try:
        raw = _git("show", f"{commit_hash}:{GPUS_JSON}")
    except subprocess.CalledProcessError:
        return None

    try:
        offerings = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(offerings, list):
        return None

    nodes = reduce_offerings_to_nodes(offerings)
    if not nodes:
        return None
    return nodes


def build_series() -> list[dict]:
    """Build the ascending-by-date list of ``{"date", "prices"}`` entries."""
    series: list[dict] = []
    for day, commit_hash in last_commit_per_day():
        prices = prices_for_commit(commit_hash)
        if prices is None:
            continue
        series.append({"date": day, "prices": prices})
    return series


def main() -> int:
    series = build_series()
    document = {
        "generated_at": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "series": series,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(document, indent=2) + "\n")

    print(f"Wrote {len(series)} daily points to {OUTPUT_PATH}")
    if series:
        print(f"Span: {series[0]['date']} -> {series[-1]['date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
