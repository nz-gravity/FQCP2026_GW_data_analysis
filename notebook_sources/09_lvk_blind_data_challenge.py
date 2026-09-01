# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Supplement: blind LVK data challenge
#
# **FQCP 2026 · Bayesian parameter estimation for gravitational-wave sources**
#
# > Google Colab worksheet. No prior Bayesian statistics or gravitational-wave
# > experience is assumed — see the
# > [glossary](https://nz-gravity.github.io/FQCP2026_GW_data_analysis/glossary.html)
# > whenever a term is new. Run from top to bottom. In the JupyterBook, **Live
# > route** cards identify the material for the session; **Extension** sections
# > may be skipped live.

# %% [markdown]
# ## Goal and route
#
# Search two synthetic detector streams for compact-binary signals, separate a detector transient from an astrophysical coincidence, estimate off-source PSDs, and run restricted parameter estimation.
#
# :::{admonition} Live route
# :class: tip
#
# Use this as a capstone after Modules 01--03, not as part of the first two-hour live route. Complete Tasks 1--7 in order and record each decision before opening instructor material.
# :::
#
#
# **Boundary:** This is a controlled classroom challenge: Gaussian line-free noise, Newtonian inspiral teaching signals, fixed response and mass ratio for PE, no calibration uncertainty, and no false-alarm-rate claim.

# %% [markdown]
# ## Rules and supplied search ranges
#
# The HDF5 file contains only H1/L1 strain and acquisition metadata. It may
# contain more than one CBC and at least one data-quality problem. The events do
# not overlap.
#
# Use two template-bank ranges:
#
# - bank A: $\mathcal M\in[16.5,26.0]M_\odot$;
# - bank B: $\mathcal M\in[26.6,41.0]M_\odot$.
#
# Use $q=0.9$ for the inexpensive search bank. For the two-dimensional PE, the
# instructor supplies a fixed mass ratio for each candidate. These search ranges
# are not detection-rate priors, and the ranking statistic below is not assigned
# a false-alarm rate.
#
# ### What you do—and what is supplied
#
# You will **choose and justify analysis settings, run the search, read trigger
# tables, diagnose the transient, define the two priors, and interpret posterior
# and residual plots**. You are not expected to derive or debug an FFT matched
# filter or MCMC implementation on your first day. The waveform, bank runner,
# coincidence bookkeeping, likelihood, sampler, and plotting scaffolds are
# provided and deliberately kept visible so you can connect the equations to the
# code.
#
# [Download the HDF5 data directly](../assets/lvk_blind_challenge.h5).

# %%
import os, sys, subprocess, importlib.util

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
missing = [
    package
    for package in ("corner", "h5py")
    if importlib.util.find_spec(package) is None
]
if missing:
    if IN_COLAB:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "corner>=2.2",
                "h5py>=3.11",
            ]
        )
    else:
        raise ImportError(
            "Install corner>=2.2 and h5py>=3.11, "
            "or use the locked workshop environment."
        )

# %%
from pathlib import Path
from urllib.request import urlretrieve

import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, spectrogram, welch

plt.style.use("seaborn-v0_8-whitegrid")

local_candidates = [
    Path("assets/lvk_blind_challenge.h5"),
    Path("../assets/lvk_blind_challenge.h5"),
]
DATA_PATH = next(
    (path for path in local_candidates if path.exists()), local_candidates[0]
)
DATA_URL = (
    "https://raw.githubusercontent.com/nz-gravity/"
    "FQCP2026_GW_data_analysis/main/assets/lvk_blind_challenge.h5"
)
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATA_URL, DATA_PATH)

with h5py.File(DATA_PATH, "r") as source:
    sampling_frequency = int(source.attrs["sampling_frequency_hz"])
    duration = float(source.attrs["duration_s"])
    start_time = float(source.attrs["start_time_s"])
    strain = {
        detector: np.asarray(source[f"strain/{detector}"][:], dtype=np.float64)
        for detector in ("H1", "L1")
    }

time = start_time + np.arange(len(strain["H1"])) / sampling_frequency
assert len(strain["H1"]) == len(strain["L1"]) == int(duration * sampling_frequency)
assert all(np.isfinite(values).all() for values in strain.values())
print(f"Loaded {duration:.0f} s at {sampling_frequency} Hz from {DATA_PATH}")
print("Detector arrays:", {name: values.shape for name, values in strain.items()})

# %% [markdown]
# ## Task 1: Inspect the data
#
# Check shapes, finite values, and detector-by-detector scale. Plot a downsampled overview, but do not expect a CBC to be visible in raw strain. **Tip:** failure to see a chirp here is expected, not evidence that the file is empty. **Suggested plot:** two aligned time-series panels with the same time axis.

# %%
# Your overview here. Keep the plot light by downsampling.
step = sampling_frequency // 8
fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
for axis, detector in zip(axes, ("H1", "L1")):
    axis.plot(time[::step], strain[detector][::step], lw=0.35)
    axis.set(ylabel=f"{detector} strain")
axes[-1].set_xlabel("time after file start [s]")
plt.show()

# %% [markdown]
# > **Record before continuing:** write one or two sentences stating
# what you observed, the decision you made, and the evidence from the output.
# Edit this Markdown cell or keep notes beside the notebook.

# %% [markdown]
# ## Task 2: Choose the analysis duration
#
# Use the lowest chirp mass in each bank and $f_\mathrm{low}=20$ Hz. Compute the leading-order inspiral time and justify an FFT-friendly analysis duration no longer than 8 seconds. **Tip:** round upward to a convenient power-of-two duration and leave room around coalescence. **Optional plot:** inspiral duration versus chirp mass across both banks.

# %%
MTSUN_SI = 4.925490947e-6


def inspiral_time(chirp_mass, f_low=20.0):
    return 5 / 256 * (MTSUN_SI * chirp_mass) ** (-5 / 3) * (np.pi * f_low) ** (-8 / 3)


for lower_edge in (16.5, 26.6):
    print(lower_edge, inspiral_time(lower_edge), "s")

analysis_duration = 8.0  # supplied default; explain why it is adequate
print("Chosen analysis duration:", analysis_duration)

mass_grid = np.linspace(16.5, 41.0, 200)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(mass_grid, inspiral_time(mass_grid))
ax.axhline(analysis_duration, color="C3", ls="--", label="chosen segment")
ax.set(xlabel=r"chirp mass [$M_\odot$]", ylabel="time from 20 Hz [s]")
ax.legend()
plt.show()

# %% [markdown]
# > **Record before continuing:** write one or two sentences stating
# what you observed, the decision you made, and the evidence from the output.
# Edit this Markdown cell or keep notes beside the notebook.

# %% [markdown]
# ### How the supplied template bank works
#
# Each template is a deliberately simplified Newtonian inspiral,
#
# $$
# \tilde h(f;\mathcal M,t_c)\propto f^{-7/6}
# \exp\left[i\left(-\frac{\pi}{4}
# +\frac{3}{128}(\pi\mathcal M_{\rm sec}f)^{-5/3}
# -2\pi f t_c\right)\right],
# $$
#
# with a smooth cutoff near the mass-dependent ISCO. The supplied bank runner:
#
# 1. chooses a grid of chirp masses;
# 2. builds one waveform for each grid point;
# 3. correlates it with each detector using the PSD-weighted matched filter;
# 4. keeps the largest SNR and corresponding template mass at every time.
#
# This is a teaching bank, not a production LVK search. On the first pass, focus
# on what goes into `run_template_bank` and what it returns; the FFT details are
# optional reading.

# %% [markdown]
# ## Task 3: Build an initial PSD and run the supplied search
#
# Matched filtering needs a PSD before the event times are known. Start with a robust median-Welch estimate from the full record. The toy waveform, matched filter, and template-bank loop are supplied below: read their inputs and outputs, then run them. Your job is to identify the common-detector peaks and explain why maximising over the bank is useful. **Suggested plot:** bank-maximum H1/L1 SNR versus time for each mass range, with an exploratory threshold at 6.

# %% jupyter={"source_hidden": true} tags=["hide-input"]
# @title Supplied machinery -- run this cell, inspect the output
WELCH_SECONDS = 4


def median_welch(values):
    return welch(
        values,
        fs=sampling_frequency,
        window=("tukey", 0.2),
        nperseg=WELCH_SECONDS * sampling_frequency,
        noverlap=WELCH_SECONDS * sampling_frequency // 2,
        detrend=False,
        average="median",
    )


initial_psd = {detector: median_welch(values) for detector, values in strain.items()}

# --- Supplied first-day search machinery ---
n_samples = len(time)
frequency = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)
frequency_spacing = 1 / duration
data_frequency = {
    detector: np.fft.rfft(values) / sampling_frequency
    for detector, values in strain.items()
}
psd_on_search_grid = {
    detector: np.interp(frequency, *initial_psd[detector]) for detector in ("H1", "L1")
}


def component_masses(chirp_mass, mass_ratio):
    eta = mass_ratio / (1 + mass_ratio) ** 2
    total_mass = chirp_mass / eta ** (3 / 5)
    primary_mass = total_mass / (1 + mass_ratio)
    return primary_mass, mass_ratio * primary_mass


def newtonian_chirp(frequency_array, chirp_mass, mass_ratio, coalescence_time=0.0):
    """Supplied toy frequency-domain inspiral with a smooth ISCO cutoff."""
    waveform = np.zeros(frequency_array.size, dtype=complex)
    primary_mass, secondary_mass = component_masses(chirp_mass, mass_ratio)
    f_isco = 1 / (6**1.5 * np.pi * MTSUN_SI * (primary_mass + secondary_mass))
    taper_start = 0.85 * f_isco
    usable = (frequency_array >= 20.0) & (frequency_array < f_isco)
    taper = np.ones(frequency_array.size)
    taper_region = (frequency_array >= taper_start) & (frequency_array < f_isco)
    taper[taper_region] = 0.5 * (
        1
        + np.cos(
            np.pi
            * (frequency_array[taper_region] - taper_start)
            / (f_isco - taper_start)
        )
    )
    phase = (
        -np.pi / 4
        + 3
        / 128
        * (np.pi * MTSUN_SI * chirp_mass * frequency_array[usable]) ** (-5 / 3)
        - 2 * np.pi * frequency_array[usable] * coalescence_time
    )
    waveform[usable] = (
        frequency_array[usable] ** (-7 / 6) * taper[usable] * np.exp(1j * phase)
    )
    return waveform


def matched_filter_snr(data_fd, template_fd, psd):
    """Return phase-maximised SNR as a function of coalescence time."""
    usable = (frequency >= 20) & (frequency <= 400) & np.isfinite(psd) & (psd > 0)
    integrand = np.zeros(frequency.size, dtype=complex)
    integrand[usable] = data_fd[usable] * np.conj(template_fd[usable]) / psd[usable]
    padded = np.zeros(n_samples, dtype=complex)
    padded[: frequency.size] = integrand
    correlation = 4 * frequency_spacing * n_samples * np.fft.ifft(padded)
    norm = np.sqrt(
        4 * frequency_spacing * np.sum(np.abs(template_fd[usable]) ** 2 / psd[usable])
    )
    return np.abs(correlation) / norm


def run_template_bank(chirp_masses, mass_ratio=0.9):
    """Maximise the SNR time series over a supplied chirp-mass grid."""
    maximum_snr = {detector: np.zeros(n_samples) for detector in ("H1", "L1")}
    maximum_mass = {detector: np.zeros(n_samples) for detector in ("H1", "L1")}
    for chirp_mass in chirp_masses:
        template = newtonian_chirp(frequency, chirp_mass, mass_ratio)
        for detector in ("H1", "L1"):
            trial = matched_filter_snr(
                data_frequency[detector], template, psd_on_search_grid[detector]
            )
            better = trial > maximum_snr[detector]
            maximum_snr[detector][better] = trial[better]
            maximum_mass[detector][better] = chirp_mass
    return maximum_snr, maximum_mass


banks = {
    "A: 16.5--26": np.linspace(16.5, 26.0, 20),
    "B: 26.6--41": np.linspace(26.6, 41.0, 24),
}
search_snr, best_mass = {}, {}
for bank_name, chirp_masses in banks.items():
    search_snr[bank_name], best_mass[bank_name] = run_template_bank(chirp_masses)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for axis, bank_name in zip(axes, banks):
    for detector in ("H1", "L1"):
        axis.plot(time, search_snr[bank_name][detector], lw=0.8, label=detector)
    axis.axhline(6, color="k", ls="--", lw=0.8)
    axis.set(ylabel="bank-max SNR", title=bank_name, xlim=(145, 235))
    axis.legend()
axes[-1].set_xlabel("time after file start [s]")
plt.show()

# %% [markdown]
# > **Record before continuing:** write one or two sentences stating
# what you observed, the decision you made, and the evidence from the output.
# Edit this Markdown cell or keep notes beside the notebook.

# %% [markdown]
# ## Task 4: Apply coincidence and handle the transient
#
# List single-detector peaks above your exploratory threshold. Pair H1 and L1 triggers within 20 ms. Inspect the unmatched loud feature in a time-frequency plot and state whether you will veto, gate, inpaint, or model it. Do not use it in a PSD estimate. The peak finder and coincidence bookkeeping are supplied; focus on reading the table. **Suggested plots:** the SNR-time plot above and matched H1/L1 spectrograms spanning ten seconds around the unmatched feature.

# %% jupyter={"source_hidden": true} tags=["hide-input"]
# @title Supplied machinery -- run this cell, inspect the output
COINCIDENCE_WINDOW = 0.020
SNR_THRESHOLD = 6.0
peak_tables = {}
coincident_candidates = []
for bank_name in banks:
    peak_tables[bank_name] = {}
    for detector in ("H1", "L1"):
        peaks, _ = find_peaks(
            search_snr[bank_name][detector],
            height=SNR_THRESHOLD,
            distance=int(0.5 * sampling_frequency),
        )
        peak_tables[bank_name][detector] = peaks
        print(f"\n{bank_name} {detector} peaks")
        for peak in peaks:
            print(
                f"  t={time[peak]:8.3f} s  "
                f"rho={search_snr[bank_name][detector][peak]:5.1f}  "
                f"Mc~{best_mass[bank_name][detector][peak]:5.1f}"
            )
    for h1_peak in peak_tables[bank_name]["H1"]:
        l1_peaks = peak_tables[bank_name]["L1"]
        separations = np.abs(time[l1_peaks] - time[h1_peak])
        if len(separations) and separations.min() <= COINCIDENCE_WINDOW:
            l1_peak = l1_peaks[np.argmin(separations)]
            candidate_time = 0.5 * (time[h1_peak] + time[l1_peak])
            coincident_candidates.append((bank_name, candidate_time, h1_peak, l1_peak))

# Cluster duplicate triggers from the two overlapping banks.
unique_candidates = []
for candidate in sorted(coincident_candidates, key=lambda item: item[1]):
    bank_name, candidate_time, h1_peak, l1_peak = candidate
    rank = np.hypot(
        search_snr[bank_name]["H1"][h1_peak],
        search_snr[bank_name]["L1"][l1_peak],
    )
    if unique_candidates and abs(candidate_time - unique_candidates[-1][1]) < 0.2:
        if rank > unique_candidates[-1][-1]:
            unique_candidates[-1] = (*candidate, rank)
    else:
        unique_candidates.append((*candidate, rank))

print("\nTime-clustered coincidences")
for bank_name, candidate_time, h1_peak, l1_peak, rank in unique_candidates:
    print(f"  {bank_name}: t~{candidate_time:.3f} s, network rank={rank:.1f}")

# The loud single-detector feature is deliberately away from either CBC.
transient_window = (time >= 185) & (time <= 195)
sos = butter(4, (20, 250), btype="bandpass", fs=sampling_frequency, output="sos")
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
for axis, detector in zip(axes, ("H1", "L1")):
    filtered = sosfiltfilt(sos, strain[detector][transient_window])
    filtered /= np.std(filtered)
    spec_frequency, spec_time, power = spectrogram(
        filtered,
        fs=sampling_frequency,
        window=("tukey", 0.2),
        nperseg=128,
        noverlap=112,
        mode="psd",
    )
    band = (spec_frequency >= 20) & (spec_frequency <= 250)
    image = axis.pcolormesh(
        185 + spec_time,
        spec_frequency[band],
        10 * np.log10(power[band] + 1e-12),
        shading="auto",
    )
    axis.set(ylabel="frequency [Hz]", title=detector)
axes[-1].set_xlabel("time after file start [s]")
fig.colorbar(image, ax=axes, label="relative power [dB]")
plt.show()

print("Write 2--3 sentences: why is the unmatched feature not a CBC candidate,")
print("and which treatment is sufficient given its separation from the signals?")

# %% [markdown]
# > **Record before continuing:** write one or two sentences stating
# what you observed, the decision you made, and the evidence from the output.
# Edit this Markdown cell or keep notes beside the notebook.

# %% [markdown]
# ## Task 5: Estimate final off-source PSDs
#
# After locating the earliest signal, choose 128 clean seconds before it. Use four-second Tukey-windowed periodograms, 50% overlap, and median averaging. The calculation is supplied; explain why this interval is off-source and count the contributing periodograms. **Suggested plot:** H1 and L1 ASD on log-log axes from 15--450 Hz.

# %%
clean = time < 128.0  # justify this only after locating the candidates
final_psd = {
    detector: median_welch(values[clean]) for detector, values in strain.items()
}
number_of_averages = 1 + (clean.sum() - WELCH_SECONDS * sampling_frequency) // (
    WELCH_SECONDS * sampling_frequency // 2
)
print("Final PSD segment length:", clean.sum() / sampling_frequency, "s")
print("Overlapping periodograms:", number_of_averages)

fig, ax = plt.subplots(figsize=(9, 4))
for detector, (frequency_welch, psd_welch) in final_psd.items():
    usable = (frequency_welch >= 15) & (frequency_welch <= 450)
    ax.loglog(frequency_welch[usable], np.sqrt(psd_welch[usable]), label=detector)
ax.set(xlabel="frequency [Hz]", ylabel=r"ASD [1/$\sqrt{\mathrm{Hz}}$]")
ax.legend()
plt.show()

# %% [markdown]
# > **Record before continuing:** write one or two sentences stating
# what you observed, the decision you made, and the evidence from the output.
# Edit this Markdown cell or keep notes beside the notebook.

# %% [markdown]
# ## Task 6: Plot the PE priors
#
# For each candidate, plot the chirp-mass and coalescence-time priors before sampling. The response, mass ratio, amplitude (hence distance), and phase are supplied and held fixed. The code below converts your coincidences into the two PE specifications. **Suggested plot:** four one-dimensional prior histograms in a 2-by-2 layout. Before running, predict which prior will be narrowed most strongly by the likelihood.

# %%
KNOWN_RESPONSE = {
    "H1": dict(gain=1.0 + 0.0j, delay=0.004),
    "L1": dict(gain=0.82 * np.exp(0.35j), delay=-0.006),
}
KNOWN_MASS_RATIOS = {"event_a": 0.85, "event_b": 1.0}
KNOWN_AMPLITUDES = {
    "event_a": 1.7273387669253173e-21,
    "event_b": 9.826371643184475e-22,
}
KNOWN_PHASES = {"event_a": 0.0, "event_b": 0.0}

ordered_candidates = sorted(unique_candidates, key=lambda candidate: candidate[1])
assert len(ordered_candidates) == 2, "Resolve the coincidence table before PE."
candidate_specs = []
for index, (bank_name, candidate_time, h1_peak, l1_peak, rank) in enumerate(
    ordered_candidates
):
    label = f"event_{'ab'[index]}"
    candidate_specs.append(
        dict(
            label=label,
            time=candidate_time,
            chirp_bounds=((16.5, 26.0), (26.6, 41.0))[index],
            mass_ratio=KNOWN_MASS_RATIOS[label],
            amplitude=KNOWN_AMPLITUDES[label],
            phase=KNOWN_PHASES[label],
            search_mass=best_mass[bank_name]["H1"][h1_peak],
        )
    )

challenge_priors = {
    spec["label"]: dict(
        chirp_mass=spec["chirp_bounds"],
        geocent_time=(spec["time"] - 0.05, spec["time"] + 0.05),
    )
    for spec in candidate_specs
}
prior_rng = np.random.default_rng(20260830)
fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
for row, spec in enumerate(candidate_specs):
    for axis, parameter in zip(axes[row], ("chirp_mass", "geocent_time")):
        bounds = challenge_priors[spec["label"]][parameter]
        axis.hist(
            prior_rng.uniform(*bounds, 4000), bins=35, density=True, histtype="step"
        )
        axis.set(xlabel=parameter, ylabel="density")
    axes[row, 0].set_title(spec["label"])
fig.suptitle("Priors before sampling")
fig.tight_layout()
plt.show()

# %% [markdown]
# > **Record before continuing:** write one or two sentences stating
# what you observed, the decision you made, and the evidence from the output.
# Edit this Markdown cell or keep notes beside the notebook.

# %% [markdown]
# ### Supplied two-parameter likelihood
#
# For fixed amplitude, phase, mass ratio, and detector response, the only varying
# parameters are $\theta=(\mathcal M,t_c)$. The supplied class evaluates the
# Gaussian log-likelihood ratio
#
# $$
# \log \Lambda(\theta)=(d|h_\theta)-\frac{1}{2}(h_\theta|h_\theta),
# \qquad
# (a|b)=4\,\mathrm{Re}\sum_f
# \frac{a^*(f)b(f)}{S_n(f)}\,\Delta f.
# $$
#
# On a first pass, treat the class as supplied machinery. Use its short interface:
#
# | Input | Meaning |
# | --- | --- |
# | `chirp_mass` | changes the inspiral phase evolution |
# | `geocent_time` | shifts the same model in time |
# | candidate specification | supplies the fixed response, mass ratio, amplitude, and phase |
#
# Locate those three inputs and the returned log likelihood; do not read or
# rewrite every implementation line. The complete class remains visible for
# students who want to inspect the FFT and detector bookkeeping later.

# %% [markdown]
# ## Task 7: Run the sampler and check the result
#
# Analyse an eight-second segment around each coincident trigger with the supplied Newtonian inspiral model, response, mass ratio, amplitude, and phase. Sample only chirp mass and coalescence time with an ordinary Gaussian-noise likelihood. The likelihood and Metropolis sampler are supplied: identify where the two sampled parameters enter, then run them. Report medians and symmetric 90% intervals. **Suggested plots:** two-chain traces, chirp-mass marginals, the chirp-mass/time joint posterior, and whitened data versus residual around each merger.

# %% jupyter={"source_hidden": true} tags=["hide-input"]
# @title Supplied machinery -- run this cell, inspect the output
PE_DURATION = 8.0


class TwoParameterCBCLikelihood:
    """Supplied Gaussian likelihood in chirp mass and geocentric time only."""

    def __init__(self, specification):
        self.parameters = {"chirp_mass": None, "geocent_time": None}
        self.specification = specification
        requested_start = specification["time"] - 6.0
        first = int(round((requested_start - start_time) * sampling_frequency))
        self.segment_start = start_time + first / sampling_frequency
        count = int(PE_DURATION * sampling_frequency)
        self.frequency = np.fft.rfftfreq(count, 1 / sampling_frequency)
        self.frequency_spacing = 1 / PE_DURATION
        self.data_frequency = {
            detector: np.fft.rfft(strain[detector][first : first + count])
            / sampling_frequency
            for detector in ("H1", "L1")
        }
        self.psd = {
            detector: np.interp(self.frequency, *final_psd[detector])
            for detector in ("H1", "L1")
        }
        self.usable = (self.frequency >= 20) & (self.frequency <= 400)

    def detector_templates(self, parameters):
        local_time = parameters["geocent_time"] - self.segment_start
        return {
            detector: self.specification["amplitude"]
            * np.exp(1j * self.specification["phase"])
            * response["gain"]
            * newtonian_chirp(
                self.frequency,
                parameters["chirp_mass"],
                self.specification["mass_ratio"],
                local_time + response["delay"],
            )
            for detector, response in KNOWN_RESPONSE.items()
        }

    def log_likelihood(self, parameters):
        log_likelihood = 0.0
        for detector, template in self.detector_templates(parameters).items():
            data = self.data_frequency[detector]
            psd = self.psd[detector]
            overlap = (
                4
                * self.frequency_spacing
                * np.real(
                    np.sum(
                        np.conj(template[self.usable])
                        * data[self.usable]
                        / psd[self.usable]
                    )
                )
            )
            norm = (
                4
                * self.frequency_spacing
                * np.sum(np.abs(template[self.usable]) ** 2 / psd[self.usable])
            )
            log_likelihood += overlap - 0.5 * norm
        return float(log_likelihood)


def log_posterior(likelihood, specification, state):
    chirp_mass, geocent_time = state
    mass_low, mass_high = specification["chirp_bounds"]
    if not (mass_low <= chirp_mass <= mass_high):
        return -np.inf
    if not (
        specification["time"] - 0.05 <= geocent_time <= specification["time"] + 0.05
    ):
        return -np.inf
    return likelihood.log_likelihood(
        dict(chirp_mass=float(chirp_mass), geocent_time=float(geocent_time))
    )


def run_metropolis(
    likelihood, specification, seed, proposal_scale, steps=6000, chains=2
):
    """Supplied random-walk sampler; returns chains and acceptance fractions."""
    sampler_rng = np.random.default_rng(20260831)
    output, acceptance = [], []
    for chain_index in range(chains):
        state = np.asarray(seed, dtype=float) + sampler_rng.normal(
            scale=0.2 * proposal_scale
        )
        current_logp = log_posterior(likelihood, specification, state)
        chain = np.empty((steps, 2))
        accepted = 0
        for step_index in range(steps):
            proposal = state + sampler_rng.normal(scale=proposal_scale)
            proposal_logp = log_posterior(likelihood, specification, proposal)
            if np.log(sampler_rng.random()) < proposal_logp - current_logp:
                state, current_logp = proposal, proposal_logp
                accepted += 1
            chain[step_index] = state
        output.append(chain)
        acceptance.append(accepted / steps)
    return output, acceptance


challenge_likelihoods = {
    spec["label"]: TwoParameterCBCLikelihood(spec) for spec in candidate_specs
}
challenge_chains, challenge_samples = {}, {}
for spec in candidate_specs:
    label = spec["label"]
    proposal_scale = np.array([0.025 if label == "event_a" else 0.06, 0.00025])
    chains, acceptance = run_metropolis(
        challenge_likelihoods[label],
        spec,
        seed=(spec["search_mass"], spec["time"]),
        proposal_scale=proposal_scale,
    )
    challenge_chains[label] = chains
    challenge_samples[label] = np.concatenate([chain[1000:] for chain in chains])
    print(label, "acceptance fractions:", np.round(acceptance, 3))

posterior_parameters = ["chirp_mass", "geocent_time"]
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for column, spec in enumerate(candidate_specs):
    label = spec["label"]
    posterior = challenge_samples[label]
    print(f"\n{label}")
    for parameter_index, parameter in enumerate(posterior_parameters):
        low, median, high = np.quantile(
            posterior[:, parameter_index], [0.05, 0.5, 0.95]
        )
        print(f"  {parameter:15s}: {median:.6f} [{low:.6f}, {high:.6f}]")
    mass_quantiles = np.quantile(posterior[:, 0], [0.05, 0.5, 0.95])
    time_offsets_ms = 1000 * (posterior[:, 1] - spec["time"])
    axes[0, column].hist(posterior[:, 0], bins=40, density=True, histtype="step")
    axes[0, column].axvspan(
        mass_quantiles[0], mass_quantiles[2], color="C0", alpha=0.15
    )
    axes[0, column].set(
        title=label, xlabel=r"$\mathcal{M}$ [$M_\odot$]", ylabel="density"
    )
    axes[1, column].hexbin(
        posterior[:, 0], time_offsets_ms, gridsize=38, mincnt=1, cmap="Blues"
    )
    axes[1, column].set(
        xlabel=r"$\mathcal{M}$ [$M_\odot$]", ylabel=r"$t_c-t_{\rm search}$ [ms]"
    )
fig.suptitle("Two-parameter conditional posteriors")
fig.tight_layout()
plt.show()

# Trace plots: both chains should overlap after burn-in and look stationary.
for spec in candidate_specs:
    fig, trace_axes = plt.subplots(2, 1, figsize=(8, 3.2), sharex=True)
    for chain in challenge_chains[spec["label"]]:
        trace_axes[0].plot(chain[:, 0], lw=0.35)
        trace_axes[1].plot(1000 * (chain[:, 1] - spec["time"]), lw=0.35)
    trace_axes[0].set(ylabel="chirp mass")
    trace_axes[1].set(xlabel="step", ylabel="time offset [ms]")
    fig.suptitle(f"{spec['label']}: convergence check")
    plt.show()

fig, residual_axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
for column, spec in enumerate(candidate_specs):
    label = spec["label"]
    posterior = challenge_samples[label]
    median_parameters = dict(zip(posterior_parameters, np.median(posterior, axis=0)))
    likelihood = challenge_likelihoods[label]
    templates = likelihood.detector_templates(median_parameters)
    for row, detector in enumerate(("H1", "L1")):
        data_fd = likelihood.data_frequency[detector]
        residual_fd = data_fd - templates[detector]
        whitened_data_fd = np.zeros_like(data_fd)
        whitened_residual_fd = np.zeros_like(residual_fd)
        whitened_data_fd[likelihood.usable] = data_fd[likelihood.usable] / np.sqrt(
            likelihood.psd[detector][likelihood.usable]
        )
        whitened_residual_fd[likelihood.usable] = residual_fd[
            likelihood.usable
        ] / np.sqrt(likelihood.psd[detector][likelihood.usable])
        count = int(PE_DURATION * sampling_frequency)
        whitened_data = np.fft.irfft(whitened_data_fd, n=count)
        whitened_residual = np.fft.irfft(whitened_residual_fd, n=count)
        scale = np.std(whitened_data[: 3 * sampling_frequency])
        local_time = (
            likelihood.segment_start
            + np.arange(count) / sampling_frequency
            - spec["time"]
        )
        residual_axes[row, column].plot(
            local_time, whitened_data / scale, color="0.65", lw=0.5, label="data"
        )
        residual_axes[row, column].plot(
            local_time,
            whitened_residual / scale,
            color="C3",
            lw=0.6,
            label="residual",
        )
        residual_axes[row, column].set(
            xlim=(-1, 0.15),
            ylabel=f"{detector} whitened",
            title=label if row == 0 else None,
        )
        residual_axes[row, column].legend(fontsize=7)
residual_axes[-1, 0].set_xlabel("time from candidate [s]")
residual_axes[-1, 1].set_xlabel("time from candidate [s]")
fig.suptitle("Posterior-median residual check")
fig.tight_layout()
plt.show()

print("Interpretation prompt: does coherent chirp-like structure remain in either")
print("residual, and what would that imply about the model?")

# %% [markdown]
# > **Record before continuing:** write one or two sentences stating
# what you observed, the decision you made, and the evidence from the output.
# Edit this Markdown cell or keep notes beside the notebook.

# %% [markdown]
# ## Submission checklist
#
# - candidate geocentric times and H1/L1 peak SNRs;
# - duration calculation from each bank's lower chirp-mass edge;
# - treatment of the unmatched transient;
# - final H1/L1 ASD plot and PSD-segment justification;
# - two-dimensional prior plot;
# - sampler acceptance fractions and a convergence comment;
# - posterior corner plots, waveform overlays, and residual checks.
#
# Do not claim an astrophysical detection probability or false-alarm rate from
# this classroom bank.
