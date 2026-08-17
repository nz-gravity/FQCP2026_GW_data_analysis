# Bayesian gravitational-wave parameter estimation

**FQCP 2026 · Beijing**

This two-hour, notebook-first mini-course follows the route from source
parameters to posterior inference: estimate a noise PSD, build a
PSD-weighted likelihood, generate compact-binary polarisations, project them
onto an LVK detector network, and introduce the LISA global-fit problem.

## Live route

| Chapter | Live time | Purpose |
| --- | ---: | --- |
| Start here | 5 min | Map from source parameters to posterior |
| The PSD and Whittle likelihood | 25 min | Estimate a PSD, whiten residuals, and build a Whittle likelihood |
| CBC physics with rippleGW | 25 min | CBC parameters and both IMRPhenomD polarisations |
| LVK response and localisation | 30 min | Antenna response, polarisation, timing localisation, PSDs, and Bilby |
| LISA response and global fitting | 35 min | LISA orbit, JaxGB TDI response, and a mini global fit |
| Questions and transitions | 15 min | Assumptions, limitations, and next steps |

The population-inference chapter is retained as self-study material. Use the
{guilabel}`rocket` button on each chapter page to open the same notebook in
Google Colab.

## Scientific boundaries

- The Whittle example assumes stationary Gaussian noise and a fixed PSD.
- rippleGW supplies a physical CBC waveform, while the live inference remains
  deliberately one-dimensional.
- The timing-only sky map is pedagogical; Bilby demonstrates detector
  projection rather than a full sampling run.
- Population measurements and selection effects are simplified.
- The LISA chapter uses a real orbit and JaxGB TDI response, but fits three
  fixed-catalogue amplitude coefficients; it is not a production global fit.

## Follow-on sources

- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [GWOSC tutorials](https://gwosc.org/tutorials/)
- [GWTC-3 population paper](https://arxiv.org/abs/2111.03634)
- [Global analysis of LISA data with GLASS](https://arxiv.org/abs/2301.03673)
