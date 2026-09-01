# Glossary and prerequisites

This course assumes no previous Bayesian statistics and no previous
gravitational-wave analysis. It does assume you can read a `for` loop and a
NumPy array. Everything else is built up in the notebooks.

Come back to this page whenever a term stops meaning anything. Each entry says
what the thing *is*, and — more usefully — what it is **not**, because most of
the confusion in this field lives in that second column.

## What you actually need before starting

| You need | You do **not** need |
| --- | --- |
| Python: variables, functions, `for` loops, slicing a NumPy array | Object-oriented Python, decorators, type hints |
| Calculus: what an integral means, roughly | Being able to do integrals by hand — the notebooks do them numerically |
| The idea of a probability distribution and a mean | Any measure theory, or a previous statistics course |
| Complex numbers: $e^{i\theta}$, magnitude, conjugate | Contour integration or complex analysis |
| A general idea that gravitational waves are ripples in spacetime | General relativity, post-Newtonian theory, numerical relativity |

If a formula looks intimidating, read the sentence under it first. Every
equation in this course is followed by prose that says what it does.

## Bayesian inference

**Prior**, $\pi(\theta)$
: What you were willing to believe about the parameters *before* looking at
  these data. It is part of the model, not a formality. *Not* a statement of
  ignorance — a "wide" prior is still a specific claim.

**Likelihood**, $\mathcal{L}(d\mid\theta)$
: How probable the observed data are *if* the parameters were $\theta$. *Not*
  the probability that $\theta$ is correct — that is the posterior, and the two
  get swapped constantly in casual speech.

**Posterior**, $p(\theta\mid d)$
: What you believe about the parameters after seeing the data. This is the
  answer a parameter-estimation analysis produces. It is a whole distribution,
  *not* a single best-fit number with an error bar attached.

**Marginalisation**
: Integrating a parameter out to get the distribution of the ones you care
  about. *Not* the same as fixing the nuisance parameter at its best value —
  that throws away its uncertainty and gives you a falsely tight answer.

**Evidence** (or marginal likelihood), $\mathcal{Z}$
: The average of the likelihood over the prior. Used to compare whole models
  against each other. It depends on the prior *by construction*, which is why a
  Bayes factor must always be quoted alongside the priors that produced it.

**Bayes factor**
: A ratio of two evidences — how much the data prefer one model over another.

**Credible interval**
: A range containing, say, 90% of the posterior probability. *Not* a confidence
  interval, which is a different (frequentist) construction that answers a
  different question.

**Posterior predictive check**
: Simulate fake datasets from your own fitted model and ask whether they look
  like the data you actually got. The main way to catch a model that is
  confidently wrong. A posterior alone can never tell you its model is wrong.

**P–P test**
: Run the whole pipeline on hundreds of simulations and check that the truth
  falls inside the 90% interval 90% of the time. This is how LVK validates
  parameter-estimation code. A single injection landing inside its interval
  proves nothing.

## Sampling

**MCMC** / **Metropolis–Hastings**
: A random walk whose visiting frequency converges to the posterior. Proposes a
  move, accepts it always if uphill and sometimes if downhill. Only needs
  likelihood *ratios*, so the unknown evidence cancels.

**Burn-in**
: The early part of a chain that still remembers where it started. Discarded,
  because it describes your starting guess rather than the posterior.

**Effective sample size** ($N_{\rm eff}$)
: How many genuinely independent samples your correlated chain is worth. This,
  not the raw chain length, sets the error on every number you quote.

**Proposal scale**
: How big a step the sampler tries. Too small and it crawls; too large and
  everything is rejected. Both failures produce a chain that looks like it ran
  fine and is wrong.

**Nested sampling**
: An algorithm that computes the evidence *and* gives posterior samples, by
  peeling the prior away from the outside in. `dynesty` and `MultiNest` are
  production implementations.

**Live points**
: The population of active samples in a nested sampler, which contracts onto
  the peak as the likelihood threshold rises.

**Fisher matrix**
: A cheap Gaussian approximation to the posterior, from the curvature of the
  likelihood at its peak. Exact for linear-Gaussian problems; only a high-SNR,
  near-linear approximation for real signals. *Not* a substitute for sampling.

**Corner plot**
: The standard display of a multi-dimensional posterior: 1D marginals on the
  diagonal, 2D marginals below.

## Noise and data

**Strain**, $h(t)$
: The fractional stretching of space a detector measures. Dimensionless, and
  fantastically small — around $10^{-21}$.

**PSD** (power spectral density), $S_n(f)$
: How the noise variance is spread across frequency. Units of strain²/Hz.
  Detector noise is much louder at some frequencies than others, and the PSD is
  how the analysis knows that.

**ASD** (amplitude spectral density)
: $\sqrt{S_n(f)}$, units strain/$\sqrt{\rm Hz}$. Just the square root of the
  PSD — plots use it because the numbers are friendlier.

**Whitening**
: Dividing each frequency component by the noise ASD, so every frequency
  carries comparable noise. This is what makes a signal visible by eye, and it
  is the same weighting the likelihood applies internally.

**Periodogram**
: The raw $|\tilde{d}(f)|^2$ estimate of the spectrum. Too noisy to use
  directly as a noise model, which is why PSDs get smoothed or fitted.

**Whittle likelihood**
: The frequency-domain Gaussian likelihood used throughout GW analysis. Assumes
  the noise is stationary and Gaussian and that Fourier bins are independent.
  Gaps, spectral lines, and non-stationarity all break those assumptions.

**Stationary**
: Statistical properties do not change with time. Real detector noise is only
  approximately stationary, and only over short stretches.

**Glitch**
: A short non-Gaussian noise transient. Looks like a signal to a naive search,
  and handling glitches is a large part of real analysis work.

**Inner product**, $(a\mid b)$
: The noise-weighted overlap of two signals,
  $4\,{\rm Re}\sum_k \tilde{a}_k\tilde{b}_k^*/S_n(f_k)\,\Delta f$. Nearly every
  GW formula is built from it.

## Signals and detection

**CBC** (compact binary coalescence)
: Two black holes or neutron stars spiralling together and merging. The source
  class behind every LVK detection so far.

**Chirp**
: The characteristic sweep upward in frequency and amplitude as a binary
  inspirals.

**Chirp mass**, $\mathcal{M}$
: The specific mass combination
  $(m_1m_2)^{3/5}/(m_1+m_2)^{1/5}$ that controls how fast the chirp sweeps. It
  is measured far more precisely than either individual mass, because the
  signal's phase depends on it directly.

**Mass ratio**, $q$
: $m_2/m_1$. Much harder to measure than the chirp mass.

**SNR** (signal-to-noise ratio)
: How loud a signal is relative to the noise. Around 8 is a marginal detection;
  GW150914 had about 24.

**Matched filter**
: Slide a template waveform through the data and measure the noise-weighted
  overlap at every arrival time. The optimal way to find a signal of known shape
  in Gaussian noise, and the basis of every CBC search.

**Template bank**
: A grid of waveforms covering the parameter space, spaced finely enough that
  any real signal is close to at least one of them.

**Waveform model / approximant**
: The theoretical prediction $h(\theta)$ — e.g. `IMRPhenomD`. Different
  approximants disagree slightly, and that disagreement is a real systematic
  error in published results.

**Antenna pattern**, $F_+, F_\times$
: How sensitive a detector is to a source in a given sky direction and
  polarisation. Detectors are not equally sensitive in all directions.

**GW150914**
: The first direct detection, September 2015. Two black holes around 36 and 29
  solar masses. Used as the real-data example in Part 2B.

## LISA

**LISA**
: A planned space-based detector: three spacecraft in a triangle 2.5 million km
  on a side, sensitive to millihertz frequencies rather than the hundreds of
  hertz LVK sees.

**TDI** (time-delay interferometry)
: The post-processing that combines delayed laser-link measurements so that
  laser frequency noise — which is enormously larger than any signal — cancels.
  Without TDI, LISA sees nothing.

**$X, Y, Z$**
: The Michelson-like TDI combinations, one per spacecraft vertex.

**$A, E, T$**
: A rotation of $X,Y,Z$ into combinations with more convenient noise
  properties. $T$ is approximately signal-insensitive at low frequency, so it is
  used as a noise monitor — approximately, *not* universally.

**Galactic binaries**
: Tens of millions of white-dwarf pairs in our galaxy, each nearly
  monochromatic. Individually resolvable ones are sources; the unresolved bulk
  becomes a noise-like foreground.

**Confusion foreground**
: The unresolved pile-up of overlapping Galactic binaries. It is a signal *and*
  part of the effective noise, which is why LISA noise models are fitted rather
  than assumed.

**MBHB** (massive black-hole binary)
: Mergers of black holes of millions of solar masses — LISA's loudest sources.

**Global fit**
: Fitting every overlapping source and the noise model *simultaneously*,
  because subtracting one source wrongly corrupts every other. This is the
  central computational problem of LISA analysis.

**Gibbs sampling**
: Update one block of parameters at a time, conditional on the current values
  of the rest. How global fits are made tractable.

**RJMCMC** (reversible-jump MCMC)
: Sampling in which the *number* of sources is itself unknown and changes during
  the run.

**Wavelet / WDM domain**
: A time–frequency representation; WDM stands for
  Wilson–Daubechies–Meyer. It is useful when the noise level or data quality
  changes with time, because affected intervals can remain localised rather
  than contaminating every ordinary Fourier bin. It does not automatically
  remove gap-edge correlations or make overlapping sources independent.

## Software

| Package | What it does here |
| --- | --- |
| `numpy` / `scipy` | Every algorithm is written in plain NumPy first |
| `bilby` | The standard LVK parameter-estimation front end |
| `dynesty` | The nested sampler Bilby drives in Part 2B |
| `corner` | Corner plots |
| `gwpy` | Fetching and handling public detector strain data |
| `rippleGW` | Fast CBC waveform generation |
| `lisatools` | LISA sensitivity curves and likelihood plumbing |
| `jaxgb` | LISA orbit and TDI response generation |
| `eryn` | Ensemble and trans-dimensional sampling for LISA-style problems |
