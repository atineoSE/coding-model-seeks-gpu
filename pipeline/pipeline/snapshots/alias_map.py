"""Model alias/rename resolution.

Builds a mapping from old model names to their current canonical names
based on known renames in the git history.
"""

from datetime import date

# Each entry: (old_name, new_name, effective_date)
# Derived from git log --oneline | grep -i rename
_RENAMES: list[tuple[str, str, date]] = [
    # 2026-02-10: Bulk rename to official marketing names (commit 67e15e4)
    ("claude-opus-4-5-20251101", "claude-opus-4-5", date(2026, 2, 10)),
    ("claude-4.5-opus", "claude-opus-4-5", date(2026, 2, 10)),
    ("claude-4.5-sonnet", "claude-sonnet-4-5", date(2026, 2, 10)),
    ("claude-4.6-opus", "claude-opus-4-6", date(2026, 2, 10)),
    ("glm-4.7", "GLM-4.7", date(2026, 2, 10)),
    ("gpt-5", "GPT-5.2", date(2026, 2, 10)),
    ("gpt-5.2", "GPT-5.2", date(2026, 2, 10)),
    ("gpt-5.2-codex", "GPT-5.2-Codex", date(2026, 2, 10)),
    ("gpt-5.2-high-reasoning", "GPT-5.2-Codex", date(2026, 2, 10)),
    ("kimi-k2-thinking", "Kimi-K2-Thinking", date(2026, 2, 10)),
    ("kimi-k2.5", "Kimi-K2.5", date(2026, 2, 10)),
    ("minimax-m2", "MiniMax-M2.1", date(2026, 2, 10)),
    ("minimax-m2.1", "MiniMax-M2.1", date(2026, 2, 10)),
    ("nemotron-3-nano", "Nemotron-3-Nano", date(2026, 2, 10)),
    ("nemotron-3-nano-30b", "Nemotron-3-Nano", date(2026, 2, 10)),
    ("deepseek-v3.2-reasoner", "DeepSeek-V3.2-Reasoner", date(2026, 2, 10)),
    ("gemini-3-flash", "Gemini-3-Flash", date(2026, 2, 10)),
    ("gemini-3-pro", "Gemini-3-Pro", date(2026, 2, 10)),
    ("gemini-3-pro-preview", "Gemini-3-Pro", date(2026, 2, 10)),
    ("qwen-3-coder", "Qwen3-Coder-480B", date(2026, 2, 10)),
    # 2026-02-11: jade-spark-2862 → Minimax-2.5 (commit ae0c67e)
    ("jade-spark-2862", "Minimax-2.5", date(2026, 2, 11)),
    # 2026-02-12: Minimax-2.5 → MiniMax-M2.5 (commit 86193e9)
    ("Minimax-2.5", "MiniMax-M2.5", date(2026, 2, 12)),
]


def resolve_model_name(name: str, snapshot_date: date) -> str:
    """Resolve a model name to its canonical form at the given snapshot date.

    Applies all renames that took effect on or before snapshot_date,
    chaining until no more renames apply.
    """
    # Build lookup: old_name → (new_name, effective_date)
    # If multiple renames for the same old name, take the earliest that applies
    changed = True
    seen = {name}
    while changed:
        changed = False
        for old, new, effective in _RENAMES:
            if name == old and snapshot_date >= effective:
                name = new
                changed = True
                if name in seen:
                    # Prevent infinite loops
                    return name
                seen.add(name)
                break
    return name
