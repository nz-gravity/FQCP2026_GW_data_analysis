# Essential literature for gravitational-wave research

This is a guided entry point to GW research. It preserves the full research workflow while keeping the reading list short. Items marked **first read** are the best starting points; other links are the next paper or tool to reach for.

:::{admonition} Use this as a reading map, not a fourth lecture
:class: tip

Follow the notebook-specific dropdowns below after the corresponding live chapter.
:::

:::{dropdown} After Notebook 00 — inference basics
Read Thrane & Talbot first; then Christensen & Meyer for a GW-specific PE
overview. Return to BayesLine/BayesWave when the PSD and likelihood assumptions
become important.
:::
:::{dropdown} After Notebook 01 — LVK compact binaries
Start with Bilby and GWOSC, then read the population material only after the
selection-effects exercise. Search papers explain how a trigger precedes PE.
:::
:::{dropdown} After Notebook 02 — LISA global fitting
Start with the LISA mission and TDI papers, then GLASS and the LISA Data
Challenge archive. The global-fit work makes most sense after the shared-
residual exercise.
:::

## Module 0: Foundational theory and fundamentals

**Focus:** baseline statistics, detection principles, and instrument physics.

### 0.1 Gravitational-wave fundamentals and detection principles

Linearised general relativity predicts the tensor polarisations $h_+$ and $h_\times$. Interferometers measure their differential arm-length response. Fabry--Perot cavities, power/signal recycling, isolation, thermal-noise reduction, and quantum squeezing are engineering responses to seismic, thermal, and quantum noise.

**Essential reading**

- **First read:** [A survey of gravitational waves](https://arxiv.org/abs/2306.03797) — modern theory-to-experiment overview.
- [Gravitational-wave detector networks](https://arxiv.org/abs/1304.0670) — response and network observing.
- [Quantum noise in gravitational-wave detectors](https://arxiv.org/abs/1503.01062) — shot noise, radiation pressure, and squeezing.

### 0.2 Bayesian inference and statistics for GWs

GW inference combines a stationary-Gaussian-noise likelihood and a prior to form a posterior. The evidence $Z$ normalises the posterior and enables model comparison; MCMC and nested sampling explore the resulting high-dimensional problem. The assumptions about noise require checking in real data.

**Essential reading**

- **First read:** [Thrane & Talbot (2018)](https://arxiv.org/abs/1809.02293) — likelihoods, priors, evidence, sampling, selection effects, and hierarchical models.
- [Christensen & Meyer (2022), Parameter estimation with gravitational waves](https://arxiv.org/abs/2204.04449).
- [MCMC for gravitational radiation data](https://arxiv.org/abs/gr-qc/0102018) (first paper on MCMC for GW)

## Module 1: The LVK operational pipeline

**Focus:** raw strain through accelerated parameter estimation.

### 1.1 Strain data, calibration, and noise estimation

Calibrated strain is distributed in frame files with data-quality information. Glitches, spectral lines, drift, and calibration uncertainty make detector noise non-Gaussian and non-stationary. Gating controls loud transients; Welch averaging estimates a PSD; BayesLine and BayesWave promote the noise model itself to an inference target.

**Essential reading and tools**

- [GWOSC](https://gwosc.org/) — strain, data products, tutorials, and posterior samples.
- **First read:** [BayesWave](https://arxiv.org/abs/1410.3835) — wavelet signal and glitch modelling.
- [BayesLine](https://arxiv.org/abs/1410.3852) — broadband PSD plus spectral lines.
- [Advanced LIGO calibration for GW150914](https://arxiv.org/abs/1602.03845).

### 1.2 Signal detection and search pipelines

CBC searches use matched filtering and template banks, with chi-squared signal-consistency tests to down-rank glitches. Unmodelled burst searches seek coherent excess power across detectors. Low-latency versions enable electromagnetic follow-up.

**Essential reading and tools**

- **First read:** [PyCBC](https://arxiv.org/abs/1705.08140).
- [CBC search in Advanced LIGO O1](https://arxiv.org/abs/1606.04856) — banks and consistency tests.
- [Coherent WaveBurst](https://arxiv.org/abs/1511.09278) — unmodelled burst searches.
- [The SPIIR online coherent pipeline](https://arxiv.org/abs/2011.06787) — low-latency modelled CBC searches.

### 1.3 Parameter-estimation frameworks

PE samples intrinsic parameters (masses, spins, tides) and extrinsic parameters (distance, inclination, sky position, time, phase), generally a 15+-dimensional posterior. LALInference established the LVK framework; Bilby and bilby_pipe provide modular current workflows.

**Essential reading and tools**

- **First read:** [Bilby](https://arxiv.org/abs/1811.02042).
- [LALInference](https://arxiv.org/abs/1409.7215).
- [bilby_pipe documentation](https://lscsoft.docs.ligo.org/bilby_pipe/).

### 1.4 PE speedup and scalability

Repeated waveform generation and likelihood evaluation dominate runtime, especially for long BNS signals. Parallel sampling, reduced-order modelling, heterodyning/relative binning, and amortised methods (variational or neural inference) address that cost. Fast results need calibration and evidence validation.

**Essential reading**

- **First read:** [Relative binning](https://arxiv.org/abs/1806.10488).
- [Relative binning in Bilby](https://arxiv.org/abs/2312.06009).
- [Reduced order quadratures](https://arxiv.org/abs/1309.7172).
- [Dingo](https://arxiv.org/abs/2111.04116) — neural posterior estimation.
- [Parallel bilby](https://arxiv.org/abs/1909.11873)

### 1.5 Waveform approximants and generators

IMRPhenom models are phenomenological frequency-domain fits; EOB models combine analytic dynamics and numerical-relativity calibration. Practical studies may require precession $\chi_p$, higher modes $(\ell,m)$, eccentricity, tides, and waveform-systematic uncertainty. NR surrogates and JAX/GPU models speed evaluation.

**Essential reading and tools**

- **First read:** [Reduced order and surrogate models for gravitational waves](https://doi.org/10.1007/s41114-022-00035-w) — a review of fast waveform representations.
- [SEOBNRv4](https://arxiv.org/abs/1611.03703).
- [NRSur7dq4](https://arxiv.org/abs/1905.09300).
- [ripple](https://ripplegw.readthedocs.io/) — JAX-based waveform tooling.

### 1.6 Event catalogues and open data

GWTC catalogues document searches, false alarms, PE, and released products. O1--O3 provide mature references; cite the specific O4 release for O4-dependent claims. Independent public-data catalogues are useful cross-checks, but use their own selection definitions.

**Essential catalogues**

- **First read:** [GWTC-1](https://arxiv.org/abs/1811.12907), [GWTC-2.1](https://arxiv.org/abs/2108.01045), [GWTC-3](https://arxiv.org/abs/2111.03606).
- [GWOSC event portal](https://gwosc.org/eventapi/).
- [IAS Open Gravitational-wave Catalog](https://gwosc.org/O3/ias/).

## Module 2: Astrophysical populations and fundamental physics

**Focus:** cosmic properties, binary demographics, and tests of GR.

### 2.1 Hierarchical population inference

Event posterior samples are combined into a model for the intrinsic population. Selection effects matter: detectable sources are not a representative draw from masses, spins, orientations, or distances. Injections estimate detection efficiency or sensitive spacetime volume. Power Law + Peak masses and $\chi_\mathrm{eff}$ distributions are useful phenomenological hyper-models.

**Essential reading**

- **First read:** [GWTC-3 compact-binary population](https://arxiv.org/abs/2111.03634).
- [Unified framework for GW populations](https://arxiv.org/abs/1809.09125).
- [GWPopulation](https://gwpopulation.readthedocs.io/).

### 2.2 Astrophysical formation channels

Compare isolated binary evolution with dynamical assembly in clusters, nuclear environments, and possibly AGN disks. Mass, spin, tilt, eccentricity, and rate evolution are population-level clues, not unique channel labels. Population-synthesis predictions and phenomenological fits must be compared selection-aware.

**Essential reading**

- [Astrophysical origin of stellar-mass BBHs](https://arxiv.org/abs/1607.05814).
- [Dynamical BBH formation in globular clusters](https://arxiv.org/abs/1601.02648).
- [Population synthesis versus GW observations](https://arxiv.org/abs/1805.08215).

### 2.3 Tests of general relativity

Tests use residuals after waveform subtraction, parametrised post-Newtonian deformations, modified propagation/dispersion, and inspiral--ringdown consistency. Claims remain conditional on noise and waveform models; multiple ringdown modes are the route to more incisive black-hole spectroscopy.

**Essential reading**

- **First read:** [Tests of GR with GWTC-3](https://arxiv.org/abs/2112.06861).
- [Tests of GR with GW150914](https://arxiv.org/abs/1602.03841).
- [Black-hole spectroscopy](https://arxiv.org/abs/1605.09286).

### 2.4 Multi-messenger astronomy and cosmology

GW170817 joined a BNS GW, gamma rays, and a kilonova, launching joint source modelling and dense-matter constraints. GW luminosity distance plus host or statistical redshift gives a standard-siren $H_0$ measurement.

**Essential reading**

- **First read:** [GW170817](https://arxiv.org/abs/1710.05832).
- [Standard-siren $H_0$ measurement](https://arxiv.org/abs/1710.05835).
- [GW170817 neutron-star radii and EoS](https://arxiv.org/abs/1805.11581).
- [Modern standard-siren review](https://arxiv.org/abs/2507.12965).
- Dark siren
- Spectral siren

## Module 3: Space-based astronomy (LISA)

**Focus:** millihertz signals, overlaps, and global PE.

### 3.1 Instrument dynamics and TDI

LISA's unequal, moving arms mean laser phase-noise cancellation is a data-analysis problem. TDI forms delayed inter-spacecraft combinations that cancel it while retaining the GW response. The moving constellation encodes sky location and polarisation.

**Essential reading**

- **First read:** [Laser Interferometer Space Antenna](https://arxiv.org/abs/1702.00786).
- [Time-delay interferometry](https://arxiv.org/abs/gr-qc/0402038).
- [Synthetic LISA](https://arxiv.org/abs/gr-qc/0509116).

### 3.2 The global-fit challenge

LISA will contain resolvable Galactic binaries, a confusion foreground, MBHBs, EMRIs, stochastic backgrounds, and instrument noise. Global fits update sources and noise together with blocked-Gibbs and reversible-jump-like ideas. The point is to fit roughly $10^4$ overlapping sources while propagating confusion and noise uncertainty.

**Essential reading and resources**

- **First read:** [GLASS global LISA analysis](https://arxiv.org/abs/2301.03673).
- [LISA Data Challenge archive](https://lisa-ldc.in2p3.fr/).
- [An efficient GPU-accelerated multi-source global-fit pipeline for LISA data analysis](https://arxiv.org/abs/2405.04690).
  

### 3.3 Source classes and astrophysics

Galactic binaries yield resolved sources plus confusion noise; EMRIs trace complex orbits near massive black holes; MBHBs probe black-hole and galaxy assembly; stochastic backgrounds can be astrophysical or cosmological.

**Essential reading**

- [LISA Astrophysics White Paper](https://www.lisamission.org/lisa-astrophysics-white-paper/) — current overview of Galactic binaries, massive black-hole binaries, and EMRIs.
- [FastEMRIWaveforms](https://arxiv.org/abs/2104.04582).
- [Science with eLISA: gravitational waves from cosmological phase transitions](https://arxiv.org/abs/1512.06239).

## Module 4: Pulsar timing arrays

**Focus:** what PTAs are, how they work, and what recent detection evidence implies.

### 4.1 PTA fundamentals

PTAs use millisecond pulsars as precision clocks. Nanohertz GWs perturb pulse arrival times and correlate residuals across pulsars. The Hellings--Downs curve is the tensor-GR prediction for this correlation; timing models, dispersion, ephemerides, and red noise must be modelled with the signal.

**Essential reading**

- **First read:** [PTA sensitivity](https://arxiv.org/abs/1301.6673).
- [Hellings--Downs correlation](https://ui.adsabs.harvard.edu/abs/1983ApJ...265L..39H/abstract).
- [Emerging PTA landscape](https://arxiv.org/abs/2603.13643).

### 4.2 Detection and current discoveries

The 2023 international PTA results found evidence for a spatially correlated nanohertz background. The next work is to measure spectrum and anisotropy, separate red noise and common processes, extend timing baselines, and seek individual continuous sources. Supermassive-black-hole binaries are the leading interpretation.

**Essential reading**

- **First read:** [NANOGrav 15-year data set](https://arxiv.org/abs/2306.16213).
- [EPTA second data release](https://arxiv.org/abs/2306.16214).
- [PPTA third data release](https://arxiv.org/abs/2306.16215).

## Module 5: Other future ground-based facilities (3G)

**Focus:** next-generation detectors and high-redshift science.

### 5.1 Next-generation detectors

Einstein Telescope and Cosmic Explorer seek major sensitivity gains, particularly below 10 Hz. ET's underground triangular/xylophone concept and CE's long-arm observatories use different architectures; cite current designs for quantitative claims.

**Essential reading**

- **First read:** [Science Case for the Einstein Telescope](https://arxiv.org/abs/1912.02622).
- [Cosmic Explorer](https://arxiv.org/abs/2109.09882).
- [Recent ET and CE overview](https://arxiv.org/abs/2505.11033).

### 5.2 3G science and high overlaps

Improved low-frequency sensitivity keeps binaries in band longer and increases the observable volume. Many long-duration signals may overlap, while population studies could trace merger demographics to $z > 10$ for favourable sources. These are design forecasts, not measurements.

**Essential reading**

- [Einstein Telescope science case](https://arxiv.org/abs/1912.02622).
- [Third-generation BNS localization and early warning](https://arxiv.org/abs/1803.09680).
- [Gravitational-wave experiments: achievements and plans](https://arxiv.org/abs/2509.25952).
