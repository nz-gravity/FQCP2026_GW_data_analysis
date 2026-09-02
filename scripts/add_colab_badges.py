"""Insert top/bottom Colab badges into each notebook_sources/*.py in course order.

Order comes from _toc.yml; only .py-backed chapters get badges (03_literature is
plain markdown and has no notebook to open). Run once; re-running is safe since
existing badge cells are replaced by marker, not duplicated.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebook_sources"
REPO = "nz-gravity/FQCP2026_GW_data_analysis"

TOP_MARKER = "<!-- colab-badge-top -->"
NEXT_MARKER = "<!-- colab-badge-next -->"


def badge(stem: str) -> str:
    url = f"https://githubtocolab.com/{REPO}/blob/main/notebooks/{stem}.ipynb"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def course_stems():
    toc = yaml.safe_load((ROOT / "_toc.yml").read_text())
    stems = []
    for part in toc["parts"]:
        for chapter in part["chapters"]:
            file = chapter["file"]
            if file.startswith("notebooks/") and (SOURCE_DIR / f"{Path(file).name}.py").exists():
                stems.append(Path(file).name)
    return stems


def strip_existing(text: str) -> str:
    for marker in (TOP_MARKER, NEXT_MARKER):
        while marker in text:
            start = text.index(marker)
            cell_start = text.rindex("\n# %% [markdown]\n", 0, start)
            end = text.index("\n# %%", start + 1) if "\n# %%" in text[start + 1 :] else len(text)
            text = text[:cell_start] + text[end:]
    return text


def insert_top(text: str, stem: str) -> str:
    header_end = text.index("\n# ---\n", text.index("# ---\n") + 1) + len("\n# ---\n")
    cell = f"\n# %% [markdown]\n# {TOP_MARKER}\n# {badge(stem)}\n"
    return text[:header_end] + cell + text[header_end:]


def append_next(text: str, next_stem: str | None) -> str:
    if next_stem is None:
        return text
    cell = (
        f"\n# %% [markdown]\n# {NEXT_MARKER}\n"
        f"# Next: {badge(next_stem)}\n"
    )
    return text.rstrip("\n") + "\n" + cell


def main():
    stems = course_stems()
    for i, stem in enumerate(stems):
        path = SOURCE_DIR / f"{stem}.py"
        text = strip_existing(path.read_text())
        text = insert_top(text, stem)
        next_stem = stems[i + 1] if i + 1 < len(stems) else None
        text = append_next(text, next_stem)
        path.write_text(text)
        print(f"{stem}: top badge, next={next_stem}")


if __name__ == "__main__":
    main()
