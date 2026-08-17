"""Small, dependency-light helpers for the FQCP 2026 teaching notebooks.

This is intentionally a single source file: Colab can download it from a pinned
GitHub tag or commit without installing a package.
"""

from __future__ import annotations

import numpy as np


def normalise_log_density(log_density, grid):
    """Normalise a one-dimensional log density on an ordered grid."""
    log_density = np.asarray(log_density, dtype=float)
    grid = np.asarray(grid, dtype=float)
    density = np.exp(log_density - np.max(log_density))
    return density / np.trapezoid(density, grid)


def equal_tailed_interval(grid, density, probability=0.9):
    """Return an equal-tailed credible interval for a grid density."""
    grid = np.asarray(grid, dtype=float)
    density = np.asarray(density, dtype=float)
    cdf = np.r_[0.0, np.cumsum((density[:-1] + density[1:]) * np.diff(grid) / 2)]
    cdf /= cdf[-1]
    tail = (1.0 - probability) / 2
    return np.interp([tail, 1.0 - tail], cdf, grid)


def frequency_inner_product(a_f, b_f, noise_psd, delta_f):
    """One-sided discrete GW inner product for positive-frequency arrays."""
    a_f = np.asarray(a_f)
    b_f = np.asarray(b_f)
    noise_psd = np.asarray(noise_psd)
    return 4 * np.real(np.sum(a_f * np.conj(b_f) / noise_psd) * delta_f)


def animation_html(animation):
    """Return an IPython HTML object that works in Colab without ffmpeg."""
    from IPython.display import HTML

    return HTML(animation.to_jshtml())
