# FQCP 2026 workshop course instructions

## Course architecture

- Organise the teaching material as three conceptual parts containing eight
  independently runnable Colab notebooks:
  1. Bayesian inference from first principles.
  2. LVK A: signals, detector response, injections, matched filtering, and a
     manual likelihood.
  3. LVK B: a direct, clearly attributed adaptation of the GWOSC GW150914
     Bilby tutorial.
  4. LVK C: event posteriors, population inference, selection effects, and PE
     checks.
  5. LISA A: source zoo, moving response, links/delays, XYZ/AET, and the roles
     of analysis codes such as Erebor/Gemoo/GLASS/Eryn.
  6. LISA B: the global-fit idea, shared residuals, Gibbs sampling, and
     Metropolis-within-Gibbs.
  7. LISA C: PSD estimation with P-splines, including uncertainty and the
     instrumental-noise/foreground identifiability boundary.
  8. LISA D: WDM time-frequency analysis, gaps, non-stationarity, and the
     limits of a diagonal WDM likelihood.
- Splitting the notebooks must improve navigation, not imply that all eight
  fit into the live session. Mark a short live route and retain extensions for
  later study.
- Every notebook must run independently from top to bottom. Do not rely on
  variables, files, or execution state created by another notebook.

## Teaching style

- Keep Part 1 Bilby-free. Teach the model, prior, likelihood, posterior,
  evidence, sampling, and checks directly in readable NumPy.
- Learn from GWOSC Tutorial 5.1: use its clear toy-model progression, manual
  prior/likelihood construction, rejection-sampling intuition, and exercises,
  without importing Bilby into the introductory notebook.
- The GW150914 notebook adapts GWOSC Tutorial 5.2. Preserve the authentic
  data -> PSD -> interferometers -> priors -> `GravitationalWaveTransient` ->
  Dynesty -> posterior flow. Clearly label its restricted non-spinning model
  and fast workshop sampler settings.
- Use the GWOSC exercise pattern throughout: a markdown question or short
  task, followed immediately by a student code cell. Keep the collapsible
  **hint** directly underneath, next to the question where it is cheap to
  reach. Solutions do not live in the lab notebook: a one-click solution gets
  read before the student has felt the problem. Each lab's solutions go in
  `notebook_sources/appendix/<stem>_answers.py`, which the build prefixes with
  the lab's own cells, so solutions are runnable and never duplicate lab text.
  Those notebooks are **instructor material**: mark them `orphan: true` and
  never link to them from a lab notebook, `_toc.yml`, or `index.md`. The
  instructor shows them; students do not find them.
- A self-check must not contain the answer. Check a property, a limiting case,
  or an independent implementation; never recompute the expression the student
  was asked to write. A starter cell carries scaffolding only -- never the
  answer, and never a `print` that merely restates the instructions.
- Explain equations and scientific assumptions before code. Use scannable
  bullets, prediction prompts, automatic checks, and hidden answers.
- Preserve explicit teaching boundaries: profiling is not marginalisation;
  toy population inference is not production hierarchical inference; fixed
  catalogues/BIC are not trans-dimensional global fits; and diagonal WDM
  likelihoods are approximations when gaps or non-stationarity induce
  covariance.

## Notebook source and generated files

- `notebook_sources/*.py` are the authoritative Jupytext `py:percent` sources.
  The matching `notebooks/*.ipynb` files are generated, output-free Colab
  artifacts that remain committed. Do not hand-edit notebook JSON.
- Make teaching changes in the relevant Jupytext source, then run
  `bash scripts/build_notebooks.sh`. The automated build is deliberately
  one-way from text sources to notebooks; do not use timestamp-based
  `jupytext --sync` in automation. Keep each source/notebook basename paired.
- Preserve cell metadata when editing. `fqcp_figure` identifies outputs for the
  reference-figure branch; `tags` and `jupyter` control hidden instructor
  cells. The hidden challenge answer retains notebook-level `orphan: true`.
- When adding, removing, or renaming a notebook, update its Jupytext source,
  generated Colab notebook, `_toc.yml`, `index.md`, README/Colab links, and any
  validation or publishing text together. The eight numbered teaching modules
  remain the core route; challenge, answer, and extension notebooks may sit
  alongside them without weakening independent execution.
- Preserve unrelated dirty work, especially changes in other notebook sources.

## Validation and publishing

- Use the locked `uv` environment. After dependency changes run `uv lock`,
  then `uv lock --check`; normally use `uv sync --locked` and
  `uv run --locked`. The project targets Python `>=3.12,<3.13`.
- Run `bash scripts/build_notebooks.sh` followed by
  `uv run --locked python validate_course.py` for a focused source/notebook
  check. Run `bash scripts/prepush_course.sh` for the complete pipeline.
- `build_site.py` consumes generated notebooks and writes `_build/html`.
  `scripts/prepush_course.sh` regenerates clean notebooks, executes them for the
  site, extracts reference figures, builds the site, then regenerates
  output-free notebooks for Git. Do not commit executed outputs.
- CI performs the cheap deterministic source/notebook validation only. Full
  execution and deployment remain local because the GW150914 module downloads
  public data and the complete run is comparatively expensive.
- Before publishing, push source and generated notebooks to `origin/main`, then
  run `bash scripts/prepush_course.sh --publish`. Colab reads from `main`; the
  command force-pushes the built site and figures to `gh-pages` and `assets`.
- Do not claim the live GW150914 Bilby notebook is fully validated unless the
  public or cached data download reaches sampling successfully.
