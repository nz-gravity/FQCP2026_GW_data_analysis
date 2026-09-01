#!/usr/bin/env bash
set -euo pipefail

for source in notebook_sources/*.py; do
    name="${source##*/}"
    uv run --locked jupytext \
        --to notebook \
        --output "notebooks/${name%.py}.ipynb" \
        "$source"
done