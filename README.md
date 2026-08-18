# FQCP 2026: Bayesian gravitational-wave parameter estimation

A Colab-first mini-course for a two-hour lecture/tutorial at the 2026 International Training Workshop on Frontiers from Quanta to Cosmos Physics.

## Course structure

The material is consolidated into three self-contained notebooks, following the teaching progression of the local `nz_bilby_cbc_workshop_2024` and `lisa_analysis_workshop` repositories.

| Notebook | Live time | Purpose |
| --- | ---: | --- |
| **`00_basics_parameter_estimation.ipynb`** | **30 min** | Prior, likelihood, posterior, evidence, MCMC, nested sampling, Whittle likelihood, PSD, and audio whitening |
| **`01_lvk_compact_binary_parameter_estimation.ipynb`** | **45 min** | CBC signals, animated detector response, matched filtering, manual network likelihood, the distance–inclination degeneracy, a full Bilby + dynesty run, and localisation |
| **`02_lisa_parameter_estimation_and_global_fit.ipynb`** | **40 min** | LISA response, manual likelihood, `AnalysisContainer`, gaps, blocked Gibbs updates, and the global fit |
| Questions/transitions | 5 min | Assumptions, limitations, next steps |

Each notebook deliberately contains more than fits the live session. Advanced
sampling, Fisher, and calibration material is marked as optional or extension
content to read afterwards.

## How the algorithms are taught

Every inference method is implemented in readable NumPy and then checked
against an independent calculation, so students can see that it works rather
than take it on faith:

| Method | Where | Independent check |
| --- | --- | --- |
| grid posterior | 00 §3 | analytic linear-Gaussian covariance |
| Metropolis–Hastings | 00 §5 | marginals overlaid on the exact grid |
| Fisher / Laplace extensions | 00 extension, 02 extension | exact for the linear model; `sigma(ln A) = 1/SNR` for LISA |
| nested sampling | 00 §6 | `log Z` agrees with the grid evidence to ~0.1 |
| blocked Gibbs | 02 §5 | posterior mean agrees with joint weighted least squares |
| P–P calibration | 00 extension | 400 simulations inside the 95% band |
| matched filtering | 01 §3 | recovered SNR matches Bilby's optimal SNR |
| full Bilby/dynesty run | 01 §7 | all four truths inside their 90% intervals |
| search + blocked Gibbs | 02 §6 | all 11 parameters recovered; residual consistent with pure noise |

## Run in Google Colab

After the repository is public, every chapter has a one-click link on the course website. Direct links have this form:

```text
https://colab.research.google.com/github/nz-gravity/FQCP2026_GW_data_analysis/blob/main/notebooks/NOTEBOOK.ipynb
```

The LVK and LISA notebooks contain pinned, conditional install cells. Python 3.12 or newer is recommended. No GPU is required.

## Reusable Colab helpers

`fqcp_helpers.py` can be fetched without publishing a package:

```python
import urllib.request
url = "https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/v1.0.0/fqcp_helpers.py"
urllib.request.urlretrieve(url, "fqcp_helpers.py")
from fqcp_helpers import frequency_inner_product
```

Use a release tag, not `main`, in released notebooks. A PyPI package is unnecessary until repeated, stable helpers justify its maintenance cost.

## Local build and validation

```bash
python -m pip install -r requirements-core.txt
python build_course.py
python validate_course.py
python build_site.py
```

`build_course.py` is the reviewable notebook source. Saved outputs are validated before deployment. `build_site.py` builds the JupyterBook into `_build/html`; the book configuration and navigation are in `_config.yml` and `_toc.yml`.

## Website deployment

`.github/workflows/pages.yml` validates notebooks, builds `_build/html` with JupyterBook, and deploys it with the official GitHub Pages actions on pushes to `main`. In repository settings, select **GitHub Actions** as the Pages source.

- Repository: `https://github.com/nz-gravity/FQCP2026_GW_data_analysis`
- Intended site: `https://nz-gravity.github.io/FQCP2026_GW_data_analysis/`

## Scientific boundaries

- The basics chapter distinguishes posterior inference from a point estimate,
  demonstrates evidence with an explicit prior-volume dependence, and states the
  assumptions behind the Whittle likelihood. The audio is an analogy, not
  detector strain converted to sound.
- rippleGW supplies a physical CBC waveform. The live LVK inference is deliberately one-dimensional; Section 7 then runs a genuine four-parameter Bilby/dynesty analysis with the same waveform.
- The timing-only sky map is pedagogical. Bilby first wraps the manual
  likelihood to verify it, and then runs a real nested-sampling analysis.
- The population section treats event masses as exact so that selection bias remains visually transparent.
- The LISA chapter uses `lisatools` sensitivity curves and `AnalysisContainer`,
  plus a real JaxGB orbit/TDI response. Its miniature global fit searches a
  chirp and three monochromatic binaries out of noise and samples them with
  blocked Gibbs, but uses plain frequency-domain templates without the
  constellation response, and holds the source count fixed. BIC enumeration is
  explicitly only a classroom proxy for RJMCMC/evidence.

## Follow-on sources

- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [GWOSC tutorials](https://gwosc.org/tutorials/)
- [Global analysis of LISA data with GLASS](https://arxiv.org/abs/2301.03673)
- [LISA Analysis Tools Workshop](https://github.com/mikekatz04/LATW)
- [LISA Data Challenge files](https://lisa-ldc.in2p3.fr/file)
- [JaxGB](https://pypi.org/project/jaxgb/)
