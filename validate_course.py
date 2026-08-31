"""Fail CI if a committed notebook is invalid, unexecuted, errored, or stale.

"Stale" means the committed notebook no longer matches what ``build_course.py``
generates.  The generator is the reviewable source, so a notebook edited by hand
would otherwise be silently reverted the next time anyone runs the build.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat

notebooks = sorted((Path(__file__).parent / "notebooks").glob("*.ipynb"))
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
        if cell.get("outputs") or cell.execution_count is not None:
            raise SystemExit(
                f"{path.name}: code cell {index} carries saved output; run "
                "build_course.py and commit the stripped notebook (the site "
                "build executes them)"
            )
    print(f"validated {path.name}")


def check_generator_is_in_sync():
    """Regenerate the notebooks in a sandbox and compare cell sources."""
    root = Path(__file__).parent
    with tempfile.TemporaryDirectory() as sandbox_name:
        sandbox = Path(sandbox_name)
        shutil.copy(root / "build_course.py", sandbox / "build_course.py")
        if (root / "assets").is_dir():
            shutil.copytree(root / "assets", sandbox / "assets")
        result = subprocess.run(
            [sys.executable, "build_course.py"],
            cwd=sandbox,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                "build_course.py failed, so the notebooks cannot be verified:\n"
                + result.stderr[-2000:]
            )
        for path in sorted((root / "notebooks").glob("*.ipynb")):
            regenerated = sandbox / "notebooks" / path.name
            if not regenerated.exists():
                raise SystemExit(f"{path.name}: build_course.py no longer writes this notebook")
            committed_cells = nbformat.read(path, as_version=4).cells
            generated_cells = nbformat.read(regenerated, as_version=4).cells
            if len(committed_cells) != len(generated_cells):
                raise SystemExit(
                    f"{path.name}: committed notebook has {len(committed_cells)} cells "
                    f"but build_course.py generates {len(generated_cells)}; "
                    "port the edit into build_course.py and rebuild"
                )
            for index, (committed, generated) in enumerate(
                zip(committed_cells, generated_cells)
            ):
                if committed.source != generated.source:
                    raise SystemExit(
                        f"{path.name}: cell {index} differs from build_course.py output. "
                        "The notebook was edited by hand; port the edit into "
                        "build_course.py and rebuild, or the change will be lost."
                    )
        print("verified notebooks match build_course.py")


check_generator_is_in_sync()
