# Bayesian gravitational-wave parameter estimation

**FQCP 2026 · Beijing**

This two-hour, notebook-first mini-course has four conceptual parts and nine
independently runnable modules: Bayesian inference, LVK analysis, LISA
analysis, and making the likelihood fast. The complete book is a reference resource; the live route selects a
small subset.

**No previous Bayesian statistics or gravitational-wave analysis is assumed.**
If you can read a `for` loop and a NumPy array, you have the prerequisites.
Start with the [glossary](glossary.md) — it lists what you need beforehand, and
defines every term the notebooks use, including what each one is commonly
mistaken for. Keep it open in a second tab.

## Live route

| Live block | Time | Modules used |
| --- | ---: | --- |
| Bayesian foundations | 30 min | Part 1 core route |
| LVK inference | 45 min | Part 2A core plus the data/PSD/likelihood path from 2B |
| LISA inference | 40 min | Part 3A response overview plus the conditional-update idea from 3B |
| Questions and transitions | 5 min | Local GWOSC-style exercises and scientific boundaries |

Each chapter contains more material than the live session covers. Advanced
sampling mechanics, Fisher, and calibration material are optional extensions
to read afterwards.

:::{admonition} One inference map for the whole course
:class: important

$$\text{data} + \text{signal model} + \text{noise model}
\longrightarrow \text{likelihood} \longrightarrow \text{posterior}
\longrightarrow \text{checks} \longrightarrow \text{claim}.$$

The notebooks progress by changing the data, response, and noise model—not by
changing Bayesian logic. Their live-route cards use JupyterBook dropdowns to separate
the in-room sequence from follow-up material.
:::

Use the {guilabel}`rocket` button on each module page to open the same
notebook in Google Colab.

The LVK supplement is a blind two-detector data challenge. Its worked answer
is built as a directly addressable page but intentionally omitted from the
visible course navigation.

## Scientific boundaries

- The basics chapter begins with Gaussian linear regression, contrasts a
  posterior with a point estimate, demonstrates evidence, and then introduces
  the stationary Gaussian Whittle model and PSD weighting as the same Gaussian
  residual calculation in the Fourier basis. A short NumPyro run samples the
  posterior already established by the transparent grid calculation.
- The optional sampler lab contains teaching implementations of random-walk
  Metropolis, nested sampling, NUTS, and Gaussian variational inference. They
  are validated against the exact grid and expose mechanics and failure modes;
  they are not substitutes for NumPyro, Stan, `dynesty`, `emcee`, or Bilby.
- Fisher-matrix material is outside the live route. Its extensions state the
  limits clearly: exact for the linear model, but only a high-SNR, near-linear
  approximation for real signals.
- rippleGW supplies a physical CBC waveform in Part 2A. Part 2B directly adapts
  the GWOSC Bilby/LAL analysis of public GW150914 data.
- The timing-only sky map is pedagogical; Bilby demonstrates detector
  projection and wraps the manual likelihood rather than running a full
  sampler.
- The population example isolates selection effects by treating event masses as
  exactly measured.
- The LISA modules use `lisatools` sensitivity curves and `AnalysisContainer`,
  plus a real JaxGB orbit/TDI response. Its gaps/non-stationarity laboratory is
  intentionally toy. Its miniature global fit does search a chirp and three
  monochromatic binaries out of noise and sample them with blocked Gibbs, but
  uses plain frequency-domain templates without the constellation response, and
  keeps the source count fixed rather than trans-dimensional.
- Part 3C uses a local Laplace approximation to visualise P-spline coefficient
  uncertainty; it is not presented as a validated production posterior.
- Part 4 builds heterodyning, relative binning, and a reduced-order model on a
  leading-order stationary-phase inspiral with an analytic toy PSD. Every
  approximation is checked against the exact likelihood in the same notebook,
  but the implementations are pedagogical rather than production ones.

## Follow-on sources

- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [GWOSC tutorials](https://gwosc.org/tutorials/)
- [GWTC-3 population paper](https://arxiv.org/abs/2111.03634)
- [Global analysis of LISA data with GLASS](https://arxiv.org/abs/2301.03673)
- [LISA Analysis Tools Workshop](https://github.com/mikekatz04/LATW)
