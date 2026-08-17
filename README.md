# FQCP 2026: Bayesian gravitational-wave parameter estimation

A Colab-first mini-course for a two-hour lecture/tutorial at the 2026 International Training Workshop on Frontiers from Quanta to Cosmos Physics.

## Course structure

The split notebooks are easier to teach and more useful as a later reference than one long notebook. The default live route omits the population chapter but preserves it for self-study.

| Notebook | Live time | Purpose |
| --- | ---: | --- |
| `00_start_here.ipynb` | 5 min | Map from source parameters to posterior |
| **`01_bayes_and_whittle.ipynb`** | **25 min** | Estimate a PSD, whiten residuals, build a Whittle likelihood |
| **`02_lvk_cbc_with_ripple.ipynb`** | **25 min** | CBC parameters and both IMRPhenomD polarisations |
| **`03_lvk_response_and_localisation.ipynb`** | **30 min** | LVK antenna response, polarisation, timing localisation, PSDs, Bilby |
| `04_population_inference.ipynb` | self-study | Mock catalogue and selection effects |
| **`05_lisa_response_and_global_fit.ipynb`** | **35 min** | LISA orbit, real JaxGB TDI responses, mini global fit |
| Questions/transitions | 15 min | Assumptions, limitations, next steps |

## Run in Google Colab

After the repository is public, every chapter has a one-click link on the course website. Direct links have this form:

```text
https://colab.research.google.com/github/nz-gravity/FQCP2026_GW_data_analysis/blob/main/notebooks/NOTEBOOK.ipynb
```

The rippleGW, Bilby, and JaxGB notebooks contain pinned, conditional install cells. Python 3.12 or newer is required by JaxGB. No GPU is required.

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

- The Whittle chapter assumes stationary Gaussian noise and a fixed PSD after its Welch estimate.
- rippleGW supplies a physical CBC waveform, but the live inference is deliberately one-dimensional.
- The timing-only sky map is pedagogical; Bilby demonstrates the full detector projection, not a full sampling run.
- Population measurements and selection are simplified.
- The LISA chapter uses a real orbit and JaxGB TDI response, but fits only three fixed-catalogue amplitude coefficients. It is not a trans-dimensional production global fit.

## Follow-on sources

- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [GWOSC tutorials](https://gwosc.org/tutorials/)
- [GWTC-3 population paper](https://arxiv.org/abs/2111.03634)
- [Global analysis of LISA data with GLASS](https://arxiv.org/abs/2301.03673)
- [LISA Data Challenge files](https://lisa-ldc.in2p3.fr/file)
- [JaxGB](https://pypi.org/project/jaxgb/)
