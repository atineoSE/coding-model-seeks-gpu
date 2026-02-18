"""Git operations for reading the OpenHands index results repository."""

import logging
import subprocess
from datetime import date
from pathlib import Path

from pipeline.snapshots.constants import SCHEMA_CONSOLIDATION_DATE

logger = logging.getLogger(__name__)


def ensure_submodule(repo_path: Path) -> None:
    """Ensure the git submodule is initialized and up to date."""
    if not (repo_path / ".git").exists():
        raise FileNotFoundError(
            f"Submodule not found at {repo_path}. " "Run: git submodule update --init --recursive"
        )


def _run_git(repo_path: Path, args: list[str]) -> str:
    """Run a git command in the given repo and return stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_dates_with_commits(repo_path: Path) -> list[date]:
    """Get all unique dates with commits since schema consolidation.

    Returns sorted list of dates (ascending).
    """
    output = _run_git(
        repo_path,
        [
            "log",
            f"--since={SCHEMA_CONSOLIDATION_DATE.isoformat()}",
            "--format=%ad",
            "--date=short",
        ],
    )
    if not output:
        return []

    dates = sorted({date.fromisoformat(d) for d in output.splitlines()})
    return dates


def get_last_commit_of_day(repo_path: Path, day: date) -> str | None:
    """Get the commit hash of the last commit on a given date.

    Returns None if no commits exist up to that date.
    """
    output = _run_git(
        repo_path,
        [
            "log",
            f"--until={day.isoformat()} 23:59:59",
            "--format=%H",
            "-1",
        ],
    )
    return output if output else None


def list_model_dirs(repo_path: Path, commit: str) -> list[str]:
    """List model directory names under results/ at a given commit."""
    try:
        output = _run_git(repo_path, ["show", f"{commit}:results/"])
    except subprocess.CalledProcessError:
        logger.warning("No results/ directory at commit %s", commit[:8])
        return []

    # git show on a tree object returns lines like:
    # "tree <commit>:results/\n\ndir1/\ndir2/\n..."
    dirs = []
    for line in output.splitlines():
        line = line.strip().rstrip("/")
        if line and not line.startswith("tree ") and line != "":
            dirs.append(line)
    return dirs


def read_file_at_commit(repo_path: Path, commit: str, path: str) -> str | None:
    """Read a file's contents at a specific commit via git show.

    Returns None if the file doesn't exist at that commit.
    """
    try:
        return _run_git(repo_path, ["show", f"{commit}:{path}"])
    except subprocess.CalledProcessError:
        return None
