"""Build a static GitHub Pages site from the executed course notebooks."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "notebooks"
SITE_DIR = ROOT / "_site"
REPOSITORY = "nz-gravity/FQCP2026_GW_data_analysis"
BRANCH = "main"

COURSE = [
    ("00_start_here", "Start here", "Course map, environment check, and vocabulary"),
    ("01_bayes_and_whittle", "Bayes and the Whittle likelihood", "Coloured noise, PSD weighting, and an animated posterior"),
    ("02_lvk_cbc_with_ripple", "LVK CBCs with rippleGW", "Physical IMRPhenomD waveforms and chirp-mass inference"),
    ("03_population_inference", "Population inference", "A detected catalogue and selection effects"),
    ("04_lisa_global_fit", "LISA global fitting", "Overlapping sources and a miniature joint fit"),
]


def colab_url(stem):
    return f"https://colab.research.google.com/github/{REPOSITORY}/blob/{BRANCH}/notebooks/{stem}.ipynb"


def navigation():
    links = "".join(f'<a href="{stem}.html">{html.escape(title)}</a>' for stem, title, _ in COURSE)
    return f'<nav class="course-nav"><a class="home" href="index.html">FQCP 2026</a>{links}</nav>'


def build_notebook_pages():
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    for stem, title, _ in COURSE:
        source = NOTEBOOK_DIR / f"{stem}.ipynb"
        notebook = nbformat.read(source, as_version=4)
        body, _ = exporter.from_notebook_node(notebook)
        body = body.replace("<title>Notebook</title>", f"<title>{html.escape(title)} · FQCP 2026</title>")
        launch = (
            f'<div class="launch"><a href="{colab_url(stem)}">'
            "Run this chapter in Google Colab</a></div>"
        )
        extra_css = """<style>
        .course-nav{position:sticky;top:0;z-index:999;display:flex;gap:1rem;align-items:center;
          padding:.75rem 1rem;background:#14213d;color:white;overflow-x:auto;box-shadow:0 2px 8px #0003}
        .course-nav a{color:#eef5ff;text-decoration:none;white-space:nowrap}.course-nav .home{font-weight:700;color:#fca311}
        .launch{max-width:960px;margin:1rem auto;padding:0 1rem}.launch a{display:inline-block;padding:.65rem 1rem;
          border-radius:.4rem;background:#fca311;color:#14213d;font-weight:700;text-decoration:none}
        </style>"""
        body = body.replace("</head>", extra_css + "</head>")
        body_start = body.find("<body")
        body_open_end = body.find(">", body_start) + 1
        if body_start < 0 or body_open_end == 0:
            raise RuntimeError(f"Could not find HTML body in {source}")
        body = body[:body_open_end] + navigation() + launch + body[body_open_end:]
        (SITE_DIR / f"{stem}.html").write_text(body, encoding="utf-8")


def build_index():
    cards = "".join(
        f'''<article><p class="number">{index:02d}</p><h2><a href="{stem}.html">{html.escape(title)}</a></h2>
        <p>{html.escape(description)}</p><p><a href="{colab_url(stem)}">Open in Colab ↗</a></p></article>'''
        for index, (stem, title, description) in enumerate(COURSE)
    )
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>FQCP 2026 · Bayesian parameter estimation</title><style>
    :root{{--ink:#14213d;--accent:#fca311;--paper:#f7f8fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);
    color:var(--ink);font:17px/1.6 system-ui,sans-serif}}header{{padding:5rem max(7vw,1rem);background:var(--ink);color:white}}
    header p{{max-width:760px;font-size:1.2rem}}h1{{font-size:clamp(2.4rem,6vw,5rem);line-height:1.02;margin:.2rem 0}}
    .eyebrow,.number{{color:var(--accent);font-weight:800;text-transform:uppercase;letter-spacing:.08em}}main{{display:grid;
    grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;padding:2rem max(7vw,1rem) 4rem}}article{{background:white;
    border-radius:.7rem;padding:1.4rem;box-shadow:0 8px 28px #14213d14}}a{{color:#0057b8}}footer{{padding:2rem max(7vw,1rem);
    border-top:1px solid #ccd3df}}code{{background:#e8ebf1;padding:.1rem .3rem;border-radius:.2rem}}
    </style></head><body><header><p class="eyebrow">FQCP 2026 · Beijing</p><h1>Bayesian parameter estimation<br>for compact binaries</h1>
    <p>A two-hour, notebook-first introduction spanning the Whittle likelihood, LVK compact binaries, population inference,
    and the idea of a LISA global fit.</p></header><main>{cards}</main><footer>Teaching materials by Avi Vajpeyi.
    Notebooks are designed for Google Colab and remain useful as a self-study reference.</footer></body></html>'''
    (SITE_DIR / "index.html").write_text(page, encoding="utf-8")


def build():
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir()
    (SITE_DIR / ".nojekyll").touch()
    build_notebook_pages()
    build_index()
    shutil.copytree(NOTEBOOK_DIR, SITE_DIR / "notebooks")
    shutil.copy2(ROOT / "fqcp_helpers.py", SITE_DIR / "fqcp_helpers.py")
    print(f"Built {len(COURSE)} chapters in {SITE_DIR}")


if __name__ == "__main__":
    build()
