"""Build deterministic, output-free Colab notebooks from Jupytext sources."""

from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebook_sources"
APPENDIX_DIR = SOURCE_DIR / "appendix"
NOTEBOOK_DIR = ROOT / "notebooks"


def build_notebook(source: Path):
    """Return the canonical ``.ipynb`` representation of one text notebook.

    An appendix source (``notebook_sources/appendix/<stem>_answers.py``) is
    prefixed with the lab notebook it answers, so solutions never duplicate the
    lab text.
    """
    notebook = jupytext.read(source)
    if source.parent == APPENDIX_DIR:
        base = SOURCE_DIR / f"{source.stem.removesuffix('_answers')}.py"
        if not base.exists():
            raise SystemExit(f"{source.name}: no lab notebook {base.name} to extend")
        notebook.cells = jupytext.read(base).cells + notebook.cells
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
    return sources + sorted(APPENDIX_DIR.glob("*.py"))


def main():
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for source in source_paths():
        destination = NOTEBOOK_DIR / f"{source.stem}.ipynb"
        nbformat.write(build_notebook(source), destination)
        print(f"wrote {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
