# FQCP 2026: Bayesian parameter estimation for compact binaries

A Colab-first, beginner-level mini-course for a two-hour lecture/tutorial at the 2026 International Training Workshop on Frontiers from Quanta to Cosmos Physics.

## Recommended structure

The material is split into short, independently useful notebooks. `00` is the landing page; the live lecture follows the bold rows.

| Notebook | Live time | Purpose |
| --- | ---: | --- |
| `00_start_here.ipynb` | 5 min | Course map, setup check, vocabulary |
| **`01_bayes_and_whittle.ipynb`** | **30 min** | Bayes' rule, PSD weighting, Whittle likelihood, posterior animation |
| **`02_lvk_cbc_with_ripple.ipynb`** | **40 min** | IMRPhenomD with rippleGW, waveform animation, chirp-mass inference |
| `03_population_inference.ipynb` | 15 min | Mock catalogue, hierarchical inference, selection effects |
| **`04_lisa_global_fit.ipynb`** | **25 min** | Overlapping long-lived signals and a miniature joint/global fit |
| Discussion and exits | 5 min | Assumptions, limitations, next steps |

This is easier to teach and much better as a later reference than one long notebook. Each notebook repeats only the few helpers it needs, so students can revisit a chapter without reconstructing hidden state.

## Run in Google Colab

Before a public repository exists, download a notebook and choose **File → Upload notebook** in [Google Colab](https://colab.research.google.com/). Once this directory is on GitHub, add one-click badges using:

```text
https://colab.research.google.com/github/nz-gravity/FQCP2026_GW_data_analysis/blob/main/notebooks/NOTEBOOK.ipynb
```

Every notebook detects Colab. The ripple chapter installs pinned `rippleGW==0.2.1`; the others use packages already available in Colab. No GPU is required.

### Reusable helpers in Colab

`fqcp_helpers.py` is a deliberately small, dependency-light module. A Colab cell can fetch and import it without publishing a PyPI package:

```python
import urllib.request

url = "https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/v1.0.0/fqcp_helpers.py"
urllib.request.urlretrieve(url, "fqcp_helpers.py")
from fqcp_helpers import frequency_inner_product
```

Use a release tag such as `v1.0.0`, not `main`, in released notebooks. That prevents future helper changes from silently altering an old worksheet.

## Run locally

```bash
python -m pip install -r requirements-core.txt
python build_course.py
jupyter lab notebooks/00_start_here.ipynb
```

## Why rippleGW?

rippleGW supplies a genuine frequency-domain CBC waveform while keeping the likelihood readable. It is JAX-based, differentiable, Colab-friendly, and lighter than making LALSuite and Bilby live-class prerequisites. The course pins `0.2.1` because rippleGW is pre-1.0 and its API may change.

- rippleGW generates `h(f; theta)`;
- the notebook constructs the PSD, inner product, simulated data, prior, likelihood, and posterior;
- Bilby is the next production-oriented step, not a live dependency.

## Should this become a PyPI package?

Not yet. A package would currently hide code students should read and add another versioned dependency. Teach the notebooks once, identify helpers that are genuinely repeated and stable, then extract a tiny package. Good eventual candidates are animation helpers, plotting style, PSD utilities, and notebook validation—not the likelihood itself.

## Build and validation

`build_course.py` is the reviewable notebook source.

```bash
python build_course.py
MPLCONFIGDIR=/tmp/fqcp-mpl python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_bayes_and_whittle.ipynb
```

The ripple notebook needs an installed `rippleGW==0.2.1`, or a Colab runtime with network access for its setup cell.

## Website deployment

`build_site.py` converts the saved, executed notebooks to static HTML and adds navigation plus Colab launch links. `.github/workflows/pages.yml` validates the notebooks, builds `_site`, uploads it as a Pages artifact, and deploys it through the official GitHub Pages actions on every push to `main`.

The intended standalone repository and site are:

- repository: `https://github.com/nz-gravity/FQCP2026_GW_data_analysis`
- website: `https://nz-gravity.github.io/FQCP2026_GW_data_analysis/`

GitHub Pages must use **GitHub Actions** as its source in the repository settings. The local parent directory is not currently a Git repository, so creating/pushing the public repository remains a separate publication step.

## Scientific boundaries

- Whittle examples assume stationary Gaussian noise and a known PSD.
- The ripple chapter uses a real waveform model but simulated single-polarisation data and a small parameter space; it is not an LVK production analysis.
- The population chapter simplifies event likelihoods and the selection function.
- The LISA chapter is a one-channel two-sinusoid analogy, not a TDI response or trans-dimensional Galactic-binary pipeline.

## Follow-on sources

- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [Bilby parameter-estimation basics](https://bilby-dev.github.io/bilby/basics-of-parameter-estimation.html)
- [Bilby CBC parameter estimation](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [GWOSC tutorials](https://gwosc.org/tutorials/)
- [GWTC-3 population paper](https://arxiv.org/abs/2111.03634)
- [LISA Analysis Tools Workshop](https://github.com/mikekatz04/LATW)
