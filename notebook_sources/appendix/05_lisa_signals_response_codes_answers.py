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
# ## Code studio: the Gaussian log likelihood

# %%
def student_lisa_log_likelihood(model, observed=template):
    residual = observed - model
    return -0.5 * inner(residual, residual)


print("perfect model:", student_lisa_log_likelihood(template))
print("zero model:   ", student_lisa_log_likelihood(np.zeros_like(template)))
print("-rho^2 / 2:   ", -0.5 * optimal_snr**2)

# %% [markdown]
# ## Question: is T a universal null channel?
#
# This one cannot run here — it needs the whole orbit-to-TDI chain rebuilt with
# `USE_BREATHING_ORBITS = True` at the top of Section 1a. Do that, rerun from
# there, and then:
#
# ```python
# print("T/A RMS ratio:", np.std(T) / np.std(A))
# ```
#
# With equal arms the ratio is at the numerical floor: T is an exact null by
# construction. With breathing arms it rises by orders of magnitude, because the
# cancellation assumed the three armlengths were identical and they no longer are.
#
# T remains a *useful* diagnostic — it is still far quieter than A and E, so
# excess power there still flags something wrong. It is not a *guaranteed* null.
# The cancellation degrades fastest at high frequency, where the light-travel time
# across the mismatch is a larger fraction of a wave period.
