"""Extract the reference figures named by build_course.py's `figure=` markers.

The notebooks in git carry no outputs.  A marked code cell is followed by a
collapsed "Expected output" block pointing at this script's output on the
force-pushed `assets` branch, so a reader who has not run the cell yet -- or
whose Colab runtime died -- can still see what it should produce.

Run against notebooks that have just been executed.
"""

import base64
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "expected"


def png_output(cell):
    """Last PNG this cell displayed, or None."""
    images = [
        output["data"]["image/png"]
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    return images[-1] if images else None


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.png"):
        stale.unlink()

    written, missing = 0, []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        for cell in nbformat.read(path, as_version=4).cells:
            slug = cell.get("metadata", {}).get("fqcp_figure")
            if not slug:
                continue
            payload = png_output(cell)
            if payload is None:
                missing.append(f"{path.name}: {slug}")
                continue
            (OUT / f"{slug}.png").write_bytes(base64.b64decode(payload))
            written += 1

    if missing:
        raise SystemExit(
            "no PNG output found for these marked cells; run the notebooks "
            "first, or drop the figure= marker:\n  " + "\n  ".join(missing)
        )
    print(f"wrote {written} reference figures to {OUT.relative_to(ROOT)}")
