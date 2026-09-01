"""Build deterministic, output-free Colab notebooks from Jupytext sources."""

from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebook_sources"
NOTEBOOK_DIR = ROOT / "notebooks"


def build_notebook(source: Path):
    """Return the canonical ``.ipynb`` representation of one text notebook."""
    notebook = jupytext.read(source)
    notebook.metadata.pop("jupytext", None)
    notebook.metadata["colab"] = {
        "name": f"{source.stem}.ipynb",
        "provenance": [],
    }
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}

    for index, cell in enumerate(notebook.cells):
        # Stable IDs keep generated-notebook diffs focused on teaching changes.
        cell.id = f"cell-{index:03d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    nbformat.validate(notebook)
    return notebook


def source_paths():
    sources = sorted(SOURCE_DIR.glob("*.py"))
    if not sources:
        raise SystemExit(f"No Jupytext sources found in {SOURCE_DIR.relative_to(ROOT)}")
    return sources


def main():
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for source in source_paths():
        destination = NOTEBOOK_DIR / f"{source.stem}.ipynb"
        nbformat.write(build_notebook(source), destination)
        print(f"wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
