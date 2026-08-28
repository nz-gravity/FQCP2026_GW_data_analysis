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

# The JupyterBook build executes every notebook, so a cell that raises fails here
# exactly as it would in CI.  Notebooks stay stripped in git.
echo "==> Building JupyterBook (executes all notebooks)"
uv run --locked python build_site.py

echo "==> Checking whitespace"
git diff --check

echo "Course is ready to push."
