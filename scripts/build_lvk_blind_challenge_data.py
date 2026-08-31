"""Build deterministic H1/L1 data for the FQCP LVK blind challenge.

The signals are controlled Newtonian stationary-phase inspirals.  They retain
the physical chirp-mass phase and coalescence-time shift, but intentionally omit
post-Newtonian corrections, merger/ringdown, spins, and a full detector response.
"""

from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "lvk_blind_challenge.h5"
SAMPLING_FREQUENCY = 1024
DURATION = 256.0
START_TIME = 0.0
MINIMUM_FREQUENCY = 20.0
MAXIMUM_FREQUENCY = 400.0
MTSUN_SI = 4.925490947e-6
NOISE_SEED = 314159
RESPONSE = {
    "H1": dict(gain=1.0 + 0.0j, delay=0.004),
    "L1": dict(gain=0.82 * np.exp(0.35j), delay=-0.006),
}


def line_free_psd(frequency, detector):
    """Smooth classroom PSD with a low-frequency wall and high-frequency rise."""
    frequency = np.asarray(frequency)
    safe_frequency = np.maximum(frequency, 1.0)
    scale = {"H1": 1.00, "L1": 1.12}[detector]
    high_frequency_knee = {"H1": 300.0, "L1": 260.0}[detector]
    shape = (
        (35.0 / safe_frequency) ** 4
        + 1.0
        + (safe_frequency / high_frequency_knee) ** 2
    )
    psd = (1e-23 * scale) ** 2 * shape
    # The challenge analyses only f >= 20 Hz.  Flatten the unused sub-band
    # spectrum instead of injecting an enormous seismic wall whose rectangular
    # eight-second leakage would dominate the classroom likelihood.
    psd[frequency < 15.0] = psd[np.searchsorted(frequency, 15.0)]
    return psd


def component_masses(chirp_mass, mass_ratio):
    eta = mass_ratio / (1 + mass_ratio) ** 2
    total_mass = chirp_mass / eta ** (3 / 5)
    primary_mass = total_mass / (1 + mass_ratio)
    return primary_mass, mass_ratio * primary_mass


def newtonian_chirp(frequency, chirp_mass, mass_ratio, coalescence_time=0.0):
    """Unit-amplitude Newtonian SPA chirp with a smooth ISCO taper."""
    frequency = np.asarray(frequency)
    waveform = np.zeros(frequency.size, dtype=complex)
    primary_mass, secondary_mass = component_masses(chirp_mass, mass_ratio)
    f_isco = 1 / (
        6 ** 1.5 * np.pi * MTSUN_SI * (primary_mass + secondary_mass)
    )
    taper_start = 0.85 * f_isco
    usable = (frequency >= MINIMUM_FREQUENCY) & (frequency < f_isco)
    taper = np.ones(frequency.size)
    taper_region = (frequency >= taper_start) & (frequency < f_isco)
    taper[taper_region] = 0.5 * (
        1
        + np.cos(
            np.pi
            * (frequency[taper_region] - taper_start)
            / (f_isco - taper_start)
        )
    )
    phase = (
        -np.pi / 4
        + 3
        / 128
        * (np.pi * MTSUN_SI * chirp_mass * frequency[usable]) ** (-5 / 3)
        - 2 * np.pi * frequency[usable] * coalescence_time
    )
    waveform[usable] = (
        frequency[usable] ** (-7 / 6) * taper[usable] * np.exp(1j * phase)
    )
    return waveform


def gaussian_noise_frequency(psd, duration, rng):
    """Draw rFFT-domain noise with E[|n(f)|^2] = PSD(f) T / 2."""
    noise = np.sqrt(psd * duration / 4) * (
        rng.normal(size=psd.size) + 1j * rng.normal(size=psd.size)
    )
    noise[0] = 0.0
    noise[-1] = 0.0
    return noise


def build_dataset():
    n_samples = int(SAMPLING_FREQUENCY * DURATION)
    frequency = np.fft.rfftfreq(n_samples, 1 / SAMPLING_FREQUENCY)
    frequency_spacing = 1 / DURATION
    rng = np.random.default_rng(NOISE_SEED)
    psds = {detector: line_free_psd(frequency, detector) for detector in RESPONSE}
    noise_frequency = {
        detector: gaussian_noise_frequency(psds[detector], DURATION, rng)
        for detector in RESPONSE
    }
    strain = {
        detector: np.fft.irfft(values * SAMPLING_FREQUENCY, n=n_samples)
        for detector, values in noise_frequency.items()
    }

    injections = [
        dict(chirp_mass=22.0, mass_ratio=0.85, time=160.0, target_snr=30.0),
        dict(chirp_mass=31.0, mass_ratio=1.00, time=224.0, target_snr=15.0),
    ]
    analysis_duration = 8.0
    analysis_count = int(analysis_duration * SAMPLING_FREQUENCY)
    analysis_frequency = np.fft.rfftfreq(
        analysis_count, 1 / SAMPLING_FREQUENCY
    )
    analysis_spacing = 1 / analysis_duration
    analysis_usable = (
        (analysis_frequency >= MINIMUM_FREQUENCY)
        & (analysis_frequency <= MAXIMUM_FREQUENCY)
    )
    for injection in injections:
        detector_templates = {}
        for detector, response in RESPONSE.items():
            detector_templates[detector] = response["gain"] * newtonian_chirp(
                analysis_frequency,
                injection["chirp_mass"],
                injection["mass_ratio"],
                6.0 + response["delay"],
            )
        reference_snr_squared = sum(
            4
            * analysis_spacing
            * np.sum(
                np.abs(detector_templates[detector][analysis_usable]) ** 2
                / line_free_psd(analysis_frequency, detector)[analysis_usable]
            )
            for detector in RESPONSE
        )
        scale = injection["target_snr"] / np.sqrt(reference_snr_squared)
        first = int((injection["time"] - 6.0) * SAMPLING_FREQUENCY)
        for detector in RESPONSE:
            signal_segment = np.fft.irfft(
                scale * detector_templates[detector] * SAMPLING_FREQUENCY,
                n=analysis_count,
            )
            strain[detector][first : first + analysis_count] += signal_segment
        injection["amplitude_scale"] = scale

    time = START_TIME + np.arange(n_samples) / SAMPLING_FREQUENCY
    glitch = np.exp(-0.5 * ((time - 190.0) / 0.035) ** 2)
    glitch *= np.cos(2 * np.pi * 75.0 * (time - 190.0))
    glitch_frequency = np.fft.rfft(glitch) / SAMPLING_FREQUENCY
    raw_glitch_snr = np.sqrt(
        4
        * frequency_spacing
        * np.sum(
            np.abs(glitch_frequency[(frequency >= 20) & (frequency <= 400)]) ** 2
            / psds["H1"][(frequency >= 20) & (frequency <= 400)]
        )
    )
    strain["H1"] += glitch * (45.0 / raw_glitch_snr)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT, "w") as target:
        target.attrs["challenge"] = "FQCP2026 LVK blind data challenge"
        target.attrs["version"] = "2"
        target.attrs["sampling_frequency_hz"] = SAMPLING_FREQUENCY
        target.attrs["duration_s"] = DURATION
        target.attrs["start_time_s"] = START_TIME
        target.attrs["strain_units"] = "dimensionless"
        target.attrs["detectors"] = "H1,L1"
        strain_group = target.create_group("strain")
        for detector in ("H1", "L1"):
            strain_group.create_dataset(
                detector,
                data=strain[detector].astype(np.float32),
                compression="gzip",
                compression_opts=6,
                shuffle=True,
            )

    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")
    for injection in injections:
        primary_mass, secondary_mass = component_masses(
            injection["chirp_mass"], injection["mass_ratio"]
        )
        print(
            f"  Mc={injection['chirp_mass']:.1f}, q={injection['mass_ratio']:.2f}, "
            f"m1={primary_mass:.2f}, m2={secondary_mass:.2f}, "
            f"network SNR={injection['target_snr']:.1f}, t={injection['time']:.1f} s"
        )
    print("  H1-only sine-Gaussian: t=190.0 s, f0=75 Hz, target SNR=45")


if __name__ == "__main__":
    build_dataset()
