# Bayesian gravitational-wave parameter estimation

**FQCP 2026 · Beijing**

This two-hour, notebook-first mini-course has three self-contained chapters:
Bayesian parameter-estimation basics, LVK compact-binary inference, and
LISA sensitivity, response, and global fitting.

## Live route

| Chapter | Live time | Purpose |
| --- | ---: | --- |
| Basics: what is PE? | 30 min | Prior, likelihood, posterior, evidence, MCMC, nested sampling, Whittle likelihood, PSD, and audio whitening |
| LVK compact binaries | 45 min | CBC signals, animated response, matched filtering, manual network likelihood, the distance–inclination degeneracy, a full Bilby run, and localisation |
| LISA and the global fit | 40 min | Moving TDI response, manual likelihood, `AnalysisContainer`, gaps, blocked Gibbs updates, and the global fit |
| Questions and transitions | 5 min | Assumptions, limitations, and next steps |

Each chapter contains more material than the live session covers. Advanced
sampling, Fisher, and calibration material is marked as optional or extension
content to read afterwards.

:::{admonition} One inference map for the whole course
:class: important

$$\text{data} + \text{signal model} + \text{noise model}
\longrightarrow \text{likelihood} \longrightarrow \text{posterior}
\longrightarrow \text{checks} \longrightarrow \text{claim}.$$

The notebooks progress by changing the data, response, and noise model—not by
changing Bayesian logic. Their live-route cards use JupyterBook tabs to separate
the in-room sequence from follow-up material.
:::

Use the {guilabel}`rocket` button on each chapter page to open the same
notebook in Google Colab.

## Scientific boundaries

- The basics chapter begins with Gaussian linear regression, contrasts a
  posterior with a point estimate, demonstrates evidence, and then introduces
  the stationary Gaussian Whittle model and PSD estimation. Its audio example
  is an analogy rather than detector strain converted to sound.
- The samplers in the basics chapter are teaching implementations: a
  random-walk Metropolis sampler and a nested sampler with MCMC-based
  constrained replacement. They are validated against the exact grid rather
  than proposed as substitutes for `dynesty`, `emcee`, or `bilby`.
- Fisher-matrix material is outside the live route. Its extensions state the
  limits clearly: exact for the linear model, but only a high-SNR, near-linear
  approximation for real signals.
- rippleGW supplies a physical CBC waveform. The live LVK posterior frees only
  chirp mass; Section 7 then runs a genuine four-parameter Bilby/dynesty
  analysis with the same waveform.
- The timing-only sky map is pedagogical; Bilby demonstrates detector
  projection and wraps the manual likelihood rather than running a full
  sampler.
- The population example isolates selection effects by treating event masses as
  exactly measured.
- The LISA chapter uses `lisatools` sensitivity curves and `AnalysisContainer`,
  plus a real JaxGB orbit/TDI response. Its gaps/non-stationarity laboratory is
  intentionally toy. Its miniature global fit does search a chirp and three
  monochromatic binaries out of noise and sample them with blocked Gibbs, but
  uses plain frequency-domain templates without the constellation response, and
  keeps the source count fixed rather than trans-dimensional.

## Follow-on sources

- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [GWOSC tutorials](https://gwosc.org/tutorials/)
- [GWTC-3 population paper](https://arxiv.org/abs/2111.03634)
- [Global analysis of LISA data with GLASS](https://arxiv.org/abs/2301.03673)
- [LISA Analysis Tools Workshop](https://github.com/mikekatz04/LATW)
