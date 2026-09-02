"""Fail CI if a committed notebook is invalid, carries outputs, or is stale.

"Stale" means a committed notebook no longer matches its Jupytext source.
Files in ``notebook_sources`` are authoritative; ``notebooks`` contains the
generated, Colab-ready copies that are committed for students.
"""

from pathlib import Path

import nbformat

from scripts.build_notebooks import build_notebook, source_paths

ROOT = Path(__file__).parent
notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
if not notebooks:
    raise SystemExit("No notebooks found")

for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            if "data:image/" in cell.source or ";base64," in cell.source:
                raise SystemExit(
                    f"{path.name}: markdown cell {index} embeds an inline image; "
                    "link to the GitHub assets branch instead"
                )
            if "\\[" in cell.source or "\\]" in cell.source:
                raise SystemExit(
                    f"{path.name}: markdown cell {index} uses \\[ or \\]; "
                    "use $$ delimiters for JupyterBook and Colab compatibility"
                )
            if cell.source.count("$$") % 2:
                raise SystemExit(
                    f"{path.name}: markdown cell {index} has an unmatched $$ delimiter"
                )
            # The notebooks are opened in Colab as often as on the site, and
            # Colab renders none of MyST's directive syntax -- it prints the
            # marker lines verbatim. Callouts are blockquotes; tabs are
            # headings. Keep index.md and glossary.md free to use MyST.
            for directive in (":::", "{tab-set}", "{tab-item}", "{admonition}"):
                if directive in cell.source:
                    raise SystemExit(
                        f"{path.name}: markdown cell {index} uses MyST-only "
                        f"syntax ({directive}); it renders as literal text in "
                        "Colab. Use a blockquote callout or a heading instead"
                    )
            control_characters = [
                character
                for character in cell.source
                if ord(character) < 32 and character != "\n"
            ]
            if control_characters:
                raise SystemExit(
                    f"{path.name}: markdown cell {index} contains control characters; "
                    "fix the corresponding file in notebook_sources"
                )
        if cell.cell_type != "code":
            continue
        if cell.get("outputs") or cell.execution_count is not None:
            raise SystemExit(
                f"{path.name}: code cell {index} carries saved output; run "
                "scripts/build_notebooks.sh and commit the stripped notebook "
                "(the site build executes them)"
            )
    print(f"validated {path.name}")


def check_sources_are_in_sync():
    """Compare every committed notebook with its canonical Jupytext build."""
    sources = source_paths()
    source_stems = {path.stem for path in sources}
    notebook_stems = {path.stem for path in notebooks}
    if source_stems != notebook_stems:
        missing_notebooks = sorted(source_stems - notebook_stems)
        missing_sources = sorted(notebook_stems - source_stems)
        details = []
        if missing_notebooks:
            details.append("missing notebooks: " + ", ".join(missing_notebooks))
        if missing_sources:
            details.append("missing Jupytext sources: " + ", ".join(missing_sources))
        raise SystemExit("source/notebook set differs; " + "; ".join(details))

    for source in sources:
        path = ROOT / "notebooks" / f"{source.stem}.ipynb"
        committed = nbformat.read(path, as_version=4)
        generated = build_notebook(source)
        if committed != generated:
            if len(committed.cells) != len(generated.cells):
                detail = (
                    f"committed notebook has {len(committed.cells)} cells but "
                    f"the source generates {len(generated.cells)}"
                )
            else:
                detail = "notebook metadata or cell content differs"
                for index, (actual, expected) in enumerate(
                    zip(committed.cells, generated.cells)
                ):
                    if actual != expected:
                        detail = f"cell {index} differs"
                        break
            raise SystemExit(
                f"{path.name}: {detail}; edit {source.relative_to(ROOT)} and run "
                "scripts/build_notebooks.sh"
            )
    print("verified notebooks match Jupytext sources")


check_sources_are_in_sync()
