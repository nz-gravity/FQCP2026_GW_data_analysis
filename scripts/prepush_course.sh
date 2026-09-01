#!/usr/bin/env bash
# Regenerate, execute, validate, and render all FQCP notebooks locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

echo "==> Checking that uv.lock matches pyproject.toml"
uv lock --check

echo "==> Synchronising the locked environment"
uv sync --locked

echo "==> Regenerating notebooks"
uv run --locked python scripts/build_lvk_blind_challenge_data.py
bash scripts/build_notebooks.sh

echo "==> Validating notebooks"
uv run --locked python validate_course.py

echo "==> Executing notebooks for the build"
for notebook in notebooks/*.ipynb; do
  uv run --locked jupyter nbconvert --execute --to notebook --inplace "$notebook" \
    --ExecutePreprocessor.timeout=900
done

# Catches a figure= marker on a cell that no longer draws anything.
echo "==> Extracting reference figures"
uv run --locked python publish_assets.py

echo "==> Building JupyterBook"
uv run --locked python build_site.py

# Outputs are for the site and the assets branch, never for a commit.
echo "==> Re-stripping notebooks"
bash scripts/build_notebooks.sh

echo "==> Re-validating clean notebooks"
uv run --locked python validate_course.py

echo "==> Checking whitespace"
git diff --check

echo "Course is ready to push."

if [ "${1:-}" != "--publish" ]; then
  echo "Re-run with --publish to deploy the site and reference figures."
  exit 0
fi

REMOTE="$(git remote get-url origin)"

# Both branches are single force-pushed commits: they carry the current build,
# never its history.  .nojekyll stops Pages from dropping Sphinx's _static and
# _images directories.
echo "==> Publishing site to gh-pages"
touch _build/html/.nojekyll
git -C _build/html init -q
git -C _build/html add -A
git -C _build/html commit -qm "Course site ($(git rev-parse --short HEAD))"
git -C _build/html push --force -q "$REMOTE" HEAD:gh-pages
rm -rf _build/html/.git

echo "==> Publishing reference figures to assets"
PUBLISH_DIR="$(mktemp -d)/expected"
mkdir -p "$PUBLISH_DIR"
cp assets/expected/*.png "$PUBLISH_DIR"
# Hand-made explainers live at the branch root, beside the expected/ figures.
cp assets/global_fit_wheel.png assets/pspline_explainer.gif "$PUBLISH_DIR/.."
git -C "$PUBLISH_DIR/.." init -q
git -C "$PUBLISH_DIR/.." add -A
git -C "$PUBLISH_DIR/.." commit -qm "Reference figures ($(git rev-parse --short HEAD))"
git -C "$PUBLISH_DIR/.." push --force -q "$REMOTE" HEAD:assets

echo "Published."
