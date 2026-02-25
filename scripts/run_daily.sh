#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git"
WORK_DIR="/workspace"

# 1. Clone the repo with submodules
git clone --recurse-submodules "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"

git config user.name "pipeline-bot"
git config user.email "pipeline-bot@users.noreply.github.com"

# 2. Update submodule to latest upstream
git -C external/openhands-index-results fetch origin main
git -C external/openhands-index-results checkout origin/main

# 3. Try to update gpuhunt to latest (best-effort; continues if already up-to-date)
pip install --no-cache-dir --upgrade gpuhunt || true

# 4. Install pipeline (deps pre-installed in image via pyproject.toml, fast editable install)
pip install --no-cache-dir -e pipeline/[pipeline]

# 5. Run the pipeline (always queries gpuhunt regardless of dependency update)
python -m pipeline.main --step all

# 6. Check for changes in data files OR submodule pointer
if git diff --quiet external/openhands-index-results web/public/data/; then
    echo "No data changes detected. Exiting."
    exit 0
fi

# 7. Commit and push
git add external/openhands-index-results web/public/data/
git commit -m "chore: update pipeline data (automated)"
git pull --rebase origin main
git push origin main
