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
#   orphan: true
# ---

# %% [markdown]
# ---
#
# # Worked solutions
#
# Everything above this line is the lab notebook, unchanged. Below are the
# solutions to its exercises, as cells that actually run: execute the notebook
# from the top and every variable they need will already exist.

# %% [markdown]
# ## Question: how tight does the tolerance need to be?

# %%
def build_binned_likelihood(tolerance):
    local_edges = bin_edges(tolerance)
    index = np.searchsorted(frequency, local_edges)
    index[-1] = frequency.size
    local_offset = frequency - np.repeat(
        0.5 * (local_edges[1:] + local_edges[:-1]), np.diff(index)
    )
    total = lambda values: np.add.reduceat(values, index[:-1])
    a0 = total(weight * data * reference_waveform.conj())
    a1 = total(weight * data * reference_waveform.conj() * local_offset)
    b0 = total(weight * np.abs(reference_waveform) ** 2)
    b1 = total(weight * np.abs(reference_waveform) ** 2 * local_offset)

    def likelihood(parameters):
        ratio = waveform(local_edges, **parameters) / waveform(local_edges, **reference)
        r0 = 0.5 * (ratio[1:] + ratio[:-1])
        r1 = (ratio[1:] - ratio[:-1]) / np.diff(local_edges)
        return (
            np.sum(a0 * r0.conj() + a1 * r1.conj()).real
            - 0.5 * np.sum(b0 * np.abs(r0) ** 2 + 2 * b1 * (r0 * r1.conj()).real)
        )

    return local_edges.size - 1, likelihood


for tolerance in tolerances:
    n_bins, likelihood = build_binned_likelihood(tolerance)
    worst = np.abs(scan_likelihood(likelihood) - exact_scan).max()
    results_by_tolerance[tolerance] = (n_bins, worst)
    print(f"tolerance {tolerance:>5}: {n_bins:>4} bins, worst dlnL = {worst:.2e}")

# %% [markdown]
# The error scales roughly as the square of the tolerance, because the neglected
# term is the quadratic one plotted in Section 3. Tightening the tolerance tenfold
# costs ten times as many bins and buys a hundredfold accuracy, so the crossing
# point is worth locating rather than guessing.
