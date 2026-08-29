#!/usr/bin/env bash
# Regenerate, execute, validate, and render the complete FQCP course locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

echo "==> Checking that uv.lock matches pyproject.toml"
uv lock --check

echo "==> Synchronising the locked environment"
uv sync --locked

echo "==> Regenerating notebooks"
uv run --locked python build_course.py

echo "==> Validating notebooks"
uv run --locked python validate_course.py

echo "==> Executing notebooks for the build"
for notebook in notebooks/*.ipynb; do
  uv run --locked jupyter nbconvert --execute --to notebook --inplace "$notebook" \
    --ExecutePreprocessor.timeout=900
done

echo "==> Building JupyterBook"
uv run --locked python build_site.py

# Catches a figure= marker on a cell that no longer draws anything.
echo "==> Extracting reference figures"
uv run --locked python publish_assets.py

# Outputs are for the site and the assets branch, never for a commit.
echo "==> Re-stripping notebooks"
uv run --locked python build_course.py

echo "==> Checking whitespace"
git diff --check

echo "Course is ready to push."
