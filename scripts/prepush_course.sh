#!/usr/bin/env bash
# Regenerate, execute, validate, and render the complete FQCP course locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_NAME="fqcp-course"

cd "$ROOT"

echo "==> Checking that uv.lock matches pyproject.toml"
uv lock --check

echo "==> Synchronising the locked environment"
uv sync --locked

echo "==> Registering the uv-managed Python kernel"
uv run --locked python -m ipykernel install --sys-prefix --name "$KERNEL_NAME" \
  --display-name "FQCP course (Python 3.12)" >/dev/null

echo "==> Regenerating notebooks"
uv run --locked python build_course.py

for notebook in \
  notebooks/00_basics_parameter_estimation.ipynb \
  notebooks/01_lvk_compact_binary_parameter_estimation.ipynb \
  notebooks/02_lisa_parameter_estimation_and_global_fit.ipynb; do
  echo "==> Executing $notebook"
  uv run --locked jupyter nbconvert --execute --to notebook --inplace "$notebook" \
    --ExecutePreprocessor.kernel_name="$KERNEL_NAME" \
    --ExecutePreprocessor.timeout=900
done

echo "==> Validating saved notebook outputs"
uv run --locked python validate_course.py

echo "==> Building JupyterBook"
uv run --locked python build_site.py

echo "==> Checking whitespace"
git diff --check

echo "Course is ready to push."
