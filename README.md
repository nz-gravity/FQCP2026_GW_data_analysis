# FQCP 2026: Bayesian gravitational-wave parameter estimation

A Colab-first mini-course for a two-hour lecture/tutorial at the 2026 International Training Workshop on Frontiers from Quanta to Cosmos Physics.

## Audience

No previous Bayesian statistics and no previous gravitational-wave analysis are
assumed. Students need to be able to read a `for` loop and a NumPy array;
everything else is built up in the notebooks. `glossary.md` collects the
statistical and gravitational-wave vocabulary in one place, with an explicit
"what this is **not**" for the terms that are routinely confused, and is the
first page of the book.

## Course structure

The core material is organised as four conceptual parts and nine independently
runnable notebooks, followed by an LVK blind-data supplement. The split follows the teaching progression of the local
`nz_bilby_cbc_workshop_2024`, `lisa_analysis_workshop`, and GWOSC parameter-
estimation tutorials without implying that every module fits into the live
two-hour session.

| Part | Notebook | Purpose |
| --- | --- | --- |
| 1 | **`01_bayesian_inference.ipynb`** | Bilby-free prior, likelihood, posterior, evidence, sampling, checks, and Whittle bridge |
| 2A | **`02_lvk_signals_injections.ipynb`** | CBC signals, detector response, injections, matched filtering, and a manual likelihood |
| 2B | **`03_lvk_gw150914_bilby.ipynb`** | Direct adaptation of the GWOSC GW150914 Bilby workflow |
| 2C | **`04_lvk_population_and_checks.ipynb`** | Population inference, selection effects, and event-level PE checks |
| 3A | **`05_lisa_signals_response_codes.ipynb`** | Source zoo, orbit/link/TDI response, likelihoods, and analysis-code map |
| 3B | **`06_lisa_global_fit_gibbs.ipynb`** | Shared residuals, global fitting, Gibbs, and Metropolis-within-Gibbs |
| 3C | **`07_lisa_pspline_psd.ipynb`** | Whittle PSD estimation with P-splines and fit diagnostics |
| 3D | **`08_lisa_wdm_time_frequency.ipynb`** | WDM time-frequency analysis, gaps, and non-stationarity |
| 4 | **`10_fast_likelihoods.ipynb`** | Heterodyning, relative binning, reduced-order modelling, and parallel/batched likelihoods |
| LVK challenge | **`09_lvk_blind_data_challenge.ipynb`** | Search, transient handling, off-source PSD estimation, and two-parameter conditional PE on hidden injections |

The worked challenge notebook,
`09_lvk_blind_data_challenge_answer.ipynb`, is built as a direct website page
but omitted from the visible JupyterBook TOC. The deterministic HDF5 fixture is
`assets/lvk_blind_challenge.h5`; regenerate it with
`scripts/build_lvk_blind_challenge_data.py`. It uses smooth line-free Gaussian
noise and an explicitly toy Newtonian inspiral; the PE fixes response, mass
ratio, amplitude/distance, and phase, and samples only chirp mass and
coalescence time.

Each notebook deliberately contains more than fits the live session. Advanced
sampling, Fisher, and calibration material is marked as optional or extension
content to read afterwards.

## How the algorithms are taught

Every inference method is implemented in readable NumPy and then checked
against an independent calculation, so students can see that it works rather
than take it on faith:

| Method | Where | Independent check |
| --- | --- | --- |
| grid posterior | 1 | analytic linear-Gaussian covariance |
| Metropolis–Hastings | 1 | marginals overlaid on the exact grid |
| Fisher / Laplace extensions | 1 and 3A | exact for the linear model; `sigma(ln A) = 1/SNR` for LISA |
| nested sampling | 1 | `log Z` agrees with the grid evidence to ~0.1 |
| blocked Gibbs | 3B | posterior mean agrees with joint weighted least squares |
| P–P calibration | 1 extension | 400 simulations inside the 95% band |
| matched filtering | 2A | recovered SNR matches Bilby's optimal SNR |
| GW150914 Bilby/Dynesty | 2B | posterior and evidence read from a genuine Bilby result |
| search + blocked Gibbs | 3B | all 11 parameters recovered; residual consistent with pure noise |
| P-spline PSD | 3C | whitened power is checked against its expected unit scale |
| heterodyning and relative binning | 4 | `dlnL` against the exact 16064-bin likelihood over a chirp-mass scan |
| reduced-order model + empirical interpolant | 4 | reconstruction of a waveform outside the training set to 1e-12 |

## Run in Google Colab

After the repository is public, every chapter has a one-click link on the course website. Direct links have this form:

```text
https://colab.research.google.com/github/nz-gravity/FQCP2026_GW_data_analysis/blob/main/notebooks/NOTEBOOK.ipynb
```

The LVK and LISA notebooks contain pinned, conditional install cells. Python
3.12 is used by the locked environment. No GPU is required.

## Local build and validation

```bash
uv sync --locked
uv run --locked python build_course.py
uv run --locked python validate_course.py
uv run --locked python build_site.py
```

`uv.lock` is the authoritative, reproducible local environment; update it deliberately with `uv lock` after changing `pyproject.toml`. `build_course.py` is the reviewable notebook source. Saved outputs are validated before deployment. `build_site.py` builds the JupyterBook into `_build/html`; the book configuration and navigation are in `_config.yml` and `_toc.yml`.

### One-command pre-push check

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
bash scripts/prepush_course.sh
```

The script checks the lockfile, synchronises the environment, regenerates and
executes all core and challenge notebooks, including the live GW150914 downloads in Part
2B, validates saved outputs, builds the
JupyterBook, and runs `git diff --check`. It needs internet access and may take
several minutes.

## Website deployment

`.github/workflows/pages.yml` validates notebooks on every push and pull
request: it checks that they are valid, carry no saved outputs, use `$$` math
delimiters, and still match `build_course.py`. It does not execute or deploy
them.

The site itself is built and force-pushed to the `gh-pages` branch by
`bash scripts/prepush_course.sh --publish`, which also publishes the reference
figures to the `assets` branch. In repository settings, select **Deploy from a
branch** and point Pages at `gh-pages`.

Publishing runs locally because executing the course takes roughly twenty
minutes and Part 2B downloads public GW150914 strain — not because of any
platform restriction. Every dependency installs on Linux.

- Repository: `https://github.com/nz-gravity/FQCP2026_GW_data_analysis`
- Intended site: `https://nz-gravity.github.io/FQCP2026_GW_data_analysis/`

## Scientific boundaries

- The basics chapter distinguishes posterior inference from a point estimate,
  demonstrates evidence with an explicit prior-volume dependence, and states the
  assumptions behind the Whittle likelihood. The audio is an analogy, not
  detector strain converted to sound.
- rippleGW supplies the physical CBC waveform in Part 2A. Part 2B instead
  follows the GWOSC Bilby/LAL workflow on public GW150914 data.
- The timing-only sky map is pedagogical. Bilby first wraps the manual
  likelihood to verify it, and then runs a real nested-sampling analysis.
- The population section treats event masses as exact so that selection bias remains visually transparent.
- The LISA modules use `lisatools` sensitivity curves and `AnalysisContainer`,
  plus a real JaxGB orbit/TDI response. Its miniature global fit searches a
  chirp and three monochromatic binaries out of noise and samples them with
  blocked Gibbs, but uses plain frequency-domain templates without the
  constellation response, and holds the source count fixed. BIC enumeration is
  explicitly only a classroom proxy for RJMCMC/evidence.
- The P-spline coefficient band is a local Laplace approximation, not a
  validated production posterior or a claim of component identifiability.
- Part 4 demonstrates the speedup algorithms on a leading-order
  stationary-phase inspiral with an analytic toy PSD, so every step stays
  readable. Each approximation is validated against the exact likelihood in the
  same notebook, but the notebook is not a substitute for Bilby's production
  relative-binning and ROQ likelihoods.

## Follow-on sources

- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [GWOSC tutorials](https://gwosc.org/tutorials/)
- [Global analysis of LISA data with GLASS](https://arxiv.org/abs/2301.03673)
- [LISA Analysis Tools Workshop](https://github.com/mikekatz04/LATW)
- [LISA Data Challenge files](https://lisa-ldc.in2p3.fr/file)
- [JaxGB](https://pypi.org/project/jaxgb/)
