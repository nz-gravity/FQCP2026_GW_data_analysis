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
        if cell.cell_type != "code":
            continue
        if cell.execution_count is None:
            raise SystemExit(f"{path.name}: code cell {index} has not been executed")
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise SystemExit(f"{path.name}: code cell {index} saved {output.ename}: {output.evalue}")
    print(f"validated {path.name}")
