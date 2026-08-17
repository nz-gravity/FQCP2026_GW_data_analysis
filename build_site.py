"""Build the JupyterBook course site from the committed, executed notebooks."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    subprocess.run(
        ["jupyter-book", "build", str(ROOT), "--path-output", str(ROOT)],
        check=True,
    )
