"""Orchestrate snapshot generation: read data, compute ranks/SOTA, return results."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import mean

from pipeline.snapshots.alias_map import resolve_model_name
from pipeline.snapshots.constants import (
    BENCHMARK_GROUP,
    BENCHMARK_GROUP_DISPLAY,
    BENCHMARK_MAP,
    DISPLAY_NAMES,
    OVERALL_NAME,
)
from pipeline.snapshots.git_repo import get_last_commit_of_day
from pipeline.snapshots.reader import read_all_models

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    """A single row in benchmarks.json."""

    model_name: str
    benchmark_name: str
    benchmark_display_name: str
    score: float
    rank: int
    cost_per_task: float | None
    benchmark_group: str = BENCHMARK_GROUP
    benchmark_group_display: str = BENCHMARK_GROUP_DISPLAY

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "benchmark_name": self.benchmark_name,
            "benchmark_display_name": self.benchmark_display_name,
            "score": self.score,
            "rank": self.rank,
            "cost_per_task": self.cost_per_task,
            "benchmark_group": self.benchmark_group,
            "benchmark_group_display": self.benchmark_group_display,
        }


@dataclass
class SotaEntry:
    """A single row in sota_scores.json."""

    benchmark_name: str
    benchmark_display_name: str
    sota_model_name: str
    sota_score: float

    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_display_name": self.benchmark_display_name,
            "sota_model_name": self.sota_model_name,
            "sota_score": self.sota_score,
        }


@dataclass
class Snapshot:
    """A complete snapshot for one date."""

    snapshot_date: date
    benchmarks: list[BenchmarkEntry] = field(default_factory=list)
    sota_scores: list[SotaEntry] = field(default_factory=list)


def generate_snapshot(repo_path: Path, snapshot_date: date) -> Snapshot | None:
    """Generate a complete snapshot for one date.

    Returns None if no commit exists for the date.
    """
    commit = get_last_commit_of_day(repo_path, snapshot_date)
    if not commit:
        logger.warning("No commit found for %s", snapshot_date)
        return None

    models = read_all_models(repo_path, commit)
    if not models:
        logger.warning("No models found at %s (commit %s)", snapshot_date, commit[:8])
        return None

    # Resolve aliases and collect per-category scores
    # Key: (resolved_model_name, benchmark_name) → (score, cost)
    # If multiple dirs resolve to the same model name, take the one with higher score
    category_scores: dict[str, list[tuple[str, float, float | None]]] = defaultdict(list)

    for model_data in models:
        resolved_name = resolve_model_name(model_data.model_name, snapshot_date)

        for score_entry in model_data.scores:
            mapped = BENCHMARK_MAP.get(score_entry.benchmark)
            if not mapped:
                continue
            bench_name, _ = mapped
            category_scores[bench_name].append(
                (resolved_name, score_entry.score, score_entry.cost_per_instance)
            )

    # Deduplicate: if same model appears multiple times in a category, keep highest score
    for bench_name in category_scores:
        entries = category_scores[bench_name]
        best: dict[str, tuple[float, float | None]] = {}
        for model, score, cost in entries:
            if model not in best or score > best[model][0]:
                best[model] = (score, cost)
        category_scores[bench_name] = [
            (model, score, cost) for model, (score, cost) in best.items()
        ]

    # Compute overall scores: only include models that have scores in ALL categories
    # so that overall is always a comparable mean across the same set of benchmarks.
    num_categories = len(category_scores)
    model_category_scores: dict[str, list[float]] = defaultdict(list)
    model_category_costs: dict[str, list[float]] = defaultdict(list)

    for _bench_name, entries in category_scores.items():
        for model, score, cost in entries:
            model_category_scores[model].append(score)
            if cost is not None:
                model_category_costs[model].append(cost)

    overall_entries: list[tuple[str, float, float | None]] = []
    for model, scores in model_category_scores.items():
        if len(scores) < num_categories:
            logger.debug(
                "Excluding %s from overall: %d/%d categories",
                model,
                len(scores),
                num_categories,
            )
            continue
        avg_score = round(mean(scores), 1)
        costs = model_category_costs.get(model, [])
        avg_cost = round(mean(costs), 2) if costs else None
        overall_entries.append((model, avg_score, avg_cost))

    category_scores[OVERALL_NAME] = overall_entries

    # Build benchmark entries with ranks
    all_benchmarks: list[BenchmarkEntry] = []
    all_sota: list[SotaEntry] = []

    # Process categories in a stable order: overall first, then alphabetical
    cat_order = [OVERALL_NAME] + sorted(k for k in category_scores if k != OVERALL_NAME)

    for bench_name in cat_order:
        entries = category_scores[bench_name]
        if not entries:
            continue

        display_name = DISPLAY_NAMES[bench_name]

        # Sort by score descending for ranking
        entries.sort(key=lambda x: x[1], reverse=True)

        # SOTA
        best_model, best_score, _ = entries[0]
        all_sota.append(
            SotaEntry(
                benchmark_name=bench_name,
                benchmark_display_name=display_name,
                sota_model_name=best_model,
                sota_score=best_score,
            )
        )

        # Ranked entries
        for rank, (model, score, cost) in enumerate(entries, 1):
            all_benchmarks.append(
                BenchmarkEntry(
                    model_name=model,
                    benchmark_name=bench_name,
                    benchmark_display_name=display_name,
                    score=score,
                    rank=rank,
                    cost_per_task=cost,
                )
            )

    logger.info(
        "Generated snapshot for %s: %d entries, %d SOTA",
        snapshot_date,
        len(all_benchmarks),
        len(all_sota),
    )

    return Snapshot(
        snapshot_date=snapshot_date,
        benchmarks=all_benchmarks,
        sota_scores=all_sota,
    )
