# Bayesian gravitational-wave parameter estimation

**FQCP 2026 · Beijing**

This two-hour, notebook-first mini-course has three self-contained chapters:
Bayesian parameter-estimation basics, LVK compact-binary inference, and
LISA sensitivity, response, and global fitting.

## Live route

| Chapter | Live time | Purpose |
| --- | ---: | --- |
| Basics: what is PE? | 30 min | Prior, likelihood, grid posterior, posterior predictive checks, and the PSD bridge |
| LVK compact binaries | 45 min | CBC parameters, rippleGW, Bilby response, injection, posterior, and localisation |
| LISA and the global fit | 40 min | Sensitivity, TDI, SNR, likelihood, overlapping sources, and source count |
| Questions and transitions | 5 min | Assumptions, limitations, and next steps |

Use the {guilabel}`rocket` button on each chapter page to open the same
notebook in Google Colab.

## Scientific boundaries

- The basics chapter begins with Gaussian linear regression and then introduces
  the stationary Gaussian Whittle model and PSD estimation.
- rippleGW supplies a physical CBC waveform, while the live LVK posterior
  deliberately frees only chirp mass.
- The timing-only sky map is pedagogical; Bilby demonstrates detector
  projection rather than a full sampling run.
- The LISA chapter uses `lisatools` sensitivity curves and a real JaxGB orbit/TDI
  response, but its global fit uses a three-source candidate catalogue and is
  not a trans-dimensional production analysis.

## Follow-on sources

- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [GWOSC tutorials](https://gwosc.org/tutorials/)
- [GWTC-3 population paper](https://arxiv.org/abs/2111.03634)
- [Global analysis of LISA data with GLASS](https://arxiv.org/abs/2301.03673)
- [LISA Analysis Tools Workshop](https://github.com/mikekatz04/LATW)
