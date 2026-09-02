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
# ## Question: how wrong is the naive estimate?

# %%
naive_error = abs(mean_grid[np.argmax(naive)] - population_mean)
corrected_error = abs(mean_grid[np.argmax(corrected)] - population_mean)
print(f"detected-catalogue mean:        {detected.mean():.3f}")
print(f"injected population mean:       {population_mean:.3f}")
print(f"naive absolute error:           {naive_error:.3f}")
print(f"selection-aware absolute error: {corrected_error:.3f}")

# %% [markdown]
# The naive estimate is biased **high**, because detectability rises with mass and
# the catalogue therefore over-represents heavy systems. It is not wrong about
# anything it was asked: it is a correct description of *the detected catalogue*.
# It is only wrong as a description of the astrophysical population, which is the
# question anyone actually wanted answered. Dividing by $\alpha(\Lambda)$ is what
# turns one question into the other.
