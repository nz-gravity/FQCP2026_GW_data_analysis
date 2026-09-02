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
# ## Question: why estimate the PSD off-source?

# %%
mask = (
    h1.strain_data.frequency_mask
    & (h1.frequency_array >= 80)
    & (h1.frequency_array <= 120)
)
band_frequency = h1.frequency_array[mask]
band_asd = h1.amplitude_spectral_density_array[mask]
near_100 = np.argmin(np.abs(band_frequency - 100))
print(f"ASD at 100 Hz / band median: {band_asd[near_100] / np.median(band_asd):.3f}")

# %% [markdown]
# The ratio is close to 1, so 100 Hz is unremarkable in this band — which is the
# point. The PSD has to describe the noise *the signal sits in*. Estimating it
# from data containing the signal would let the signal inflate its own assumed
# noise floor and quietly suppress itself, exactly the Section-2 failure of
# notebook 01 with the sign reversed.

# %% [markdown]
# ## Question: comparing with the published posterior

# %%
for label, values in [
    ("GWTC-2.1", published["chirp_mass"]),
    ("this notebook", chirp_mass_samples),
]:
    low, median, high = np.quantile(values, [0.05, 0.5, 0.95])
    print(f"{label:15s} Mc = {median:.2f} [{low:.2f}, {high:.2f}]  width {high-low:.2f}")

print(f"\nour prior: U({prior['chirp_mass'].minimum}, {prior['chirp_mass'].maximum})")
inside = np.mean(
    (published["chirp_mass"] > prior["chirp_mass"].minimum)
    & (published["chirp_mass"] < prior["chirp_mass"].maximum)
)
print(f"fraction of the published posterior our prior even allows: {inside:.1%}")

# %% [markdown]
# ### 1. Do they agree?
#
# It depends which test you mean, and that is the first thing worth saying out
# loud.
#
# - **Medians:** yes, near enough. Ours sits about $0.4\,M_\odot$ high, well
#   inside the published 90% interval.
# - **Intervals:** ours is *contained in* theirs, which is not the same as
#   agreeing. Our 90% interval is roughly four times narrower.
# - **Widths:** no. We are far more confident than the LVK, on the same event,
#   from the same data.
#
# Being more precise than the published analysis, using less of everything, is
# not a success. It is the notebook-01 lesson in the wild: a narrow posterior
# reports what the model was told, not what the data know.
#
# ### 2. Why?
#
# Everything below differs between the two analyses:
#
# | | this notebook | GWTC-2.1 |
# | --- | --- | --- |
# | waveform | restricted, non-spinning | IMRPhenomXPHM + SEOBNRv4PHM, precessing |
# | spins | fixed at zero | 6 spin parameters sampled |
# | extrinsic | sky and orientation fixed | fully sampled |
# | $\mathcal M$ prior | $U(30, 32.5)$ | wide, astrophysical |
# | calibration | ignored | marginalised |
# | PSD | our own short estimate | on-source, carefully validated |
# | sampler | minutes, few hundred samples | hours to days, converged |
#
# Most of those widen a posterior. The one that dominates here is visible in the
# left panel: **our prior is narrower than the published posterior.** The dotted
# lines cut straight through the grey histogram — the printout above shows what
# fraction of the published posterior our prior even permits. A parameter cannot
# be measured to be outside its prior, so part of our apparent precision was
# assumed, not inferred.
#
# The rest comes from fixing spins and extrinsic parameters. Every parameter
# held fixed is a direction the posterior cannot spread into, and chirp mass is
# correlated with several of them.
#
# ### 3. What would you change?
#
# **Widen the chirp-mass prior**, to something like $U(25, 40)$, and rerun.
# Nothing else is worth trying first, because until the prior stops binding you
# cannot see what the data alone would have said.
#
# How you would know it worked: the posterior should get *wider* and its edges
# should stop coinciding with the prior edges. If it widens to roughly the
# published width, the prior was the whole story. If it widens only a little,
# the fixed spins and extrinsic parameters are carrying the rest — and that is a
# waveform-cost problem, not a prior problem, and no amount of sampling fixes
# it.
#
# Note that "run the sampler longer" is a different kind of fix, and not the one
# needed here. More samples would reduce the noise on our estimate of a posterior
# that is the wrong width. Convergence and correctness are separate problems.
