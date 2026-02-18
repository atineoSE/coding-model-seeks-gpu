"""Parse metadata.json and scores.json from git show output."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.snapshots.git_repo import list_model_dirs, read_file_at_commit

logger = logging.getLogger(__name__)


@dataclass
class ScoreEntry:
    """A single benchmark score for a model."""

    benchmark: str  # repo benchmark name (e.g. "swe-bench")
    score: float
    cost_per_instance: float | None = None


@dataclass
class ModelData:
    """All data for one model at one commit."""

    model_name: str  # from metadata.json "model" field
    scores: list[ScoreEntry] = field(default_factory=list)


def read_model_data(
    repo_path: Path,
    commit: str,
    dir_name: str,
) -> ModelData | None:
    """Read metadata.json and scores.json for one model directory at a commit.

    Returns None if metadata or scores can't be parsed.
    """
    base = f"results/{dir_name}"

    meta_raw = read_file_at_commit(repo_path, commit, f"{base}/metadata.json")
    if meta_raw is None:
        return None

    try:
        meta = json.loads(meta_raw)
    except json.JSONDecodeError:
        logger.warning("Invalid metadata.json for %s at %s", dir_name, commit[:8])
        return None

    model_name = meta.get("model", "").strip()
    if not model_name:
        logger.warning("No model name in metadata for %s at %s", dir_name, commit[:8])
        return None

    scores_raw = read_file_at_commit(repo_path, commit, f"{base}/scores.json")
    if scores_raw is None:
        return ModelData(model_name=model_name)

    try:
        scores_list = json.loads(scores_raw)
    except json.JSONDecodeError:
        logger.warning("Invalid scores.json for %s at %s", dir_name, commit[:8])
        return ModelData(model_name=model_name)

    entries = []
    for s in scores_list:
        benchmark = s.get("benchmark", "").strip()
        score = s.get("score")
        if not benchmark or score is None:
            continue

        # cost_per_instance was added later; earlier data has total_cost (skip it)
        cost = s.get("cost_per_instance")

        entries.append(
            ScoreEntry(
                benchmark=benchmark,
                score=float(score),
                cost_per_instance=float(cost) if cost is not None else None,
            )
        )

    return ModelData(model_name=model_name, scores=entries)


def read_all_models(
    repo_path: Path,
    commit: str,
) -> list[ModelData]:
    """Read all model data at a given commit."""
    dirs = list_model_dirs(repo_path, commit)
    models = []
    for d in dirs:
        data = read_model_data(repo_path, commit, d)
        if data is not None:
            models.append(data)
    return models
