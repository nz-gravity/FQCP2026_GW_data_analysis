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
# <!-- colab-badge-top -->
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/nz-gravity/FQCP2026_GW_data_analysis/blob/main/notebooks/09_lvk_blind_data_challenge.ipynb)

# %% [markdown]
# # Supplement: blind LVK data challenge
#
# **FQCP 2026 · Bayesian parameter estimation for gravitational-wave sources**
#
# %% [markdown]
# ## The brief
#
# Two synthetic detector streams. Find what is in them.
#
# This is not a guided exercise, and there is no prescribed method. You get the
# data, the signal model, and a short list of things that are true. How you go
# from there to an answer is yours to decide.
#
# ### What is true
#
# - Two detectors, H1 and L1, sampled continuously. No gaps.
# - The noise is Gaussian, stationary, and free of spectral lines.
# - There is **at least one binary black hole**, with chirp mass somewhere in
#   $\mathcal M\in[16.5,\,41.0]\,M_\odot$.
# - There is **at least one glitch**: a loud transient that is *not* an
#   astrophysical signal.
# - Nothing overlaps in time. Whatever is in there, it happens one thing at a
#   time.
#
# Note the phrasing. "At least one" is not "exactly one".
#
# ### What to hand in
#
# 1. **Times.** Every signal and every glitch you find, with its time. Say which
#    is which, and say how you decided.
# 2. **A chirp mass with an uncertainty**, for each candidate you fit —
#    including the glitch. A number without an error bar is not an answer.
# 3. **Anything else you found interesting.** This one is not filler. The most
#    interesting thing in the data may not be on the list above.
#
# ### Suggestions, not instructions
#
# If you want a starting point:
#
# - Look at the data first. A spectrogram of whitened strain will tell you a
#   lot before you fit anything.
# - Estimate a PSD from data *away* from whatever you find, then redo your
#   analysis with it.
# - Matched filtering is the standard tool and the toolbox below has what you
#   need for it. It is not the only option — excess power, a Q-transform,
#   band-passed energy, or plain cross-correlation between the two detectors
#   will all find a loud transient.
# - Two detectors are better than one. Ask what a real astrophysical signal
#   does in both that a local instrumental artefact does not.
# - When you fit, fit for an uncertainty, not a point estimate. Notebook 01's
#   grid is entirely sufficient for two parameters.
#
# ### One honest warning
#
# You will be tempted to fit a CBC waveform to everything you find and report
# the chirp mass. Do that — it is item 2 on the list. Then look hard at what the
# fit to the glitch is telling you, and at whether your analysis had any way of
# knowing it was wrong. Notebook 01 spent an hour on exactly this failure.
#
# > **Boundary**
# >
# > A controlled classroom challenge: Gaussian line-free noise, Newtonian inspiral
# > teaching signals, a fixed detector response, no calibration uncertainty. Do
# > not quote a false-alarm rate or an astrophysical detection probability from
# > it.
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
# %% [markdown]
# ## The signal model
#
# A compact-binary inspiral in the frequency domain, to leading (Newtonian)
# order:
#
# $$
# \tilde h(f;\mathcal M,t_c)\propto f^{-7/6}
# \exp\left[i\left(-\frac{\pi}{4}
# +\frac{3}{128}(\pi\mathcal M_{\rm sec}f)^{-5/3}
# -2\pi f t_c\right)\right],
# $$
#
# tapered smoothly to zero near the innermost stable circular orbit. The chirp
# mass sets how fast the frequency sweeps; $t_c$ slides the whole thing in time.
# This is a teaching waveform: no post-Newtonian corrections, no merger or
# ringdown, no spins, no precession.

# %%
MTSUN_SI = 4.925490947e-6  # solar mass in seconds, G M_sun / c^3


def component_masses(chirp_mass, mass_ratio):
    eta = mass_ratio / (1 + mass_ratio) ** 2
    total_mass = chirp_mass / eta ** (3 / 5)
    primary_mass = total_mass / (1 + mass_ratio)
    return primary_mass, mass_ratio * primary_mass


def newtonian_chirp(frequency_array, chirp_mass, mass_ratio=0.9, coalescence_time=0.0):
    """Frequency-domain inspiral with a smooth ISCO taper."""
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


def inspiral_duration(chirp_mass, f_low=20.0):
    """Seconds from f_low to coalescence — use it to size your analysis segments."""
    return 5 / 256 * (MTSUN_SI * chirp_mass) ** (-5 / 3) * (np.pi * f_low) ** (-8 / 3)


print(f"a {16.5:.1f} Msun chirp lasts {inspiral_duration(16.5):.1f} s from 20 Hz")
print(f"a {41.0:.1f} Msun chirp lasts {inspiral_duration(41.0):.1f} s from 20 Hz")

# %% [markdown]
# ## Toolbox
#
# A PSD estimator and a matched filter, because neither is the point of the
# exercise and both are tedious to debug. Use them, replace them, or ignore
# them.

# %%
def median_welch(values, seconds=4):
    """Median-averaged Welch PSD: robust to the transients we are looking for."""
    return welch(
        values,
        fs=sampling_frequency,
        window=("tukey", 0.2),
        nperseg=seconds * sampling_frequency,
        noverlap=seconds * sampling_frequency // 2,
        detrend=False,
        average="median",
    )


frequency = np.fft.rfftfreq(time.size, 1 / sampling_frequency)
frequency_spacing = 1 / duration


def matched_filter_snr(data_fd, template_fd, psd, f_min=20.0, f_max=400.0):
    """Phase-maximised SNR as a function of coalescence time."""
    usable = (
        (frequency >= f_min) & (frequency <= f_max) & np.isfinite(psd) & (psd > 0)
    )
    integrand = np.zeros(frequency.size, dtype=complex)
    integrand[usable] = data_fd[usable] * np.conj(template_fd[usable]) / psd[usable]
    padded = np.zeros(time.size, dtype=complex)
    padded[: frequency.size] = integrand
    correlation = 4 * frequency_spacing * time.size * np.fft.ifft(padded)
    norm = np.sqrt(
        4 * frequency_spacing * np.sum(np.abs(template_fd[usable]) ** 2 / psd[usable])
    )
    return np.abs(correlation) / norm


def to_frequency_domain(values):
    return np.fft.rfft(values) / sampling_frequency


print("available:", "median_welch, newtonian_chirp, matched_filter_snr,")
print("           inspiral_duration, to_frequency_domain")

# %% [markdown]
# ## Your workspace
#
# Everything below is yours. Add as many cells as you need.

# %%
# Start here.

# %% [markdown]
# ## Before you stop
#
# Write your three answers somewhere you can read them out:
#
# 1. the times, and which are signals and which are glitches;
# 2. a chirp mass and an uncertainty for each thing you fitted;
# 3. the interesting thing.
#
# And one question worth answering for yourself: which of your conclusions
# would survive if the noise were not Gaussian?

# %% [markdown]
# ## Resources, if you get stuck
#
# Use these for methods, not for the hidden answer:
#
# - The [GWOSC Open Data Workshop](https://learn.gwosc.org/courses/odw2025)
#   covers conditioning, matched filtering, detector characterisation, and a
#   separate public data challenge.
# - [Bilby's compact-binary tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
#   shows what a fuller signal model and likelihood would add to this two-parameter fit.
# - The course [reading map](03_literature.md) points to search pipelines,
#   BayesWave/BayesLine, and public catalogues without revealing this fixture.

# %% [markdown]
# <!-- colab-badge-next -->
# Next: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/nz-gravity/FQCP2026_GW_data_analysis/blob/main/notebooks/05_lisa_signals_response_codes.ipynb)
