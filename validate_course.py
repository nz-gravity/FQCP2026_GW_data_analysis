"""Fail CI if a committed course notebook is invalid, unexecuted, or errored."""

from pathlib import Path
import nbformat

notebooks = sorted((Path(__file__).parent / "notebooks").glob("*.ipynb"))
if not notebooks:
    raise SystemExit("No notebooks found")

for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            if "\\[" in cell.source or "\\]" in cell.source:
                raise SystemExit(
                    f"{path.name}: markdown cell {index} uses \\[ or \\]; "
                    "use $$ delimiters for JupyterBook and Colab compatibility"
                )
            if cell.source.count("$$") % 2:
                raise SystemExit(
                    f"{path.name}: markdown cell {index} has an unmatched $$ delimiter"
                )
            control_characters = [
                character
                for character in cell.source
                if ord(character) < 32 and character != "\n"
            ]
            if control_characters:
                raise SystemExit(
                    f"{path.name}: markdown cell {index} contains control characters; "
                    "use a raw string for LaTeX in build_course.py"
                )
        if cell.cell_type != "code":
            continue
        if cell.execution_count is None:
            raise SystemExit(f"{path.name}: code cell {index} has not been executed")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise SystemExit(
                    f"{path.name}: code cell {index} saved {output.ename}: {output.evalue}"
                )
    print(f"validated {path.name}")
