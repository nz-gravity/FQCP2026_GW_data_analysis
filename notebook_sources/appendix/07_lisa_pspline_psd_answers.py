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
# ## Question: what does the penalty control?

# %%
for penalty in penalties:
    result = minimize(
        lambda beta: pspline_objective(beta, penalty), fit.x, method="L-BFGS-B"
    )
    fits_by_penalty[penalty] = np.exp(basis @ result.x)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
for penalty, estimate in fits_by_penalty.items():
    ax.loglog(frequency, estimate, label=f"$\\lambda$ = {penalty:g}")
ax.loglog(frequency, true_psd, "k--", lw=2, label="truth")
ax.set(xlabel="frequency [Hz]", ylabel="PSD", title="The penalty sets the bias-variance trade")
ax.legend(fontsize=8)
plt.show()

# %% [markdown]
# $\lambda=2$ is too weak: the fit chases individual periodogram spikes, which are
# noise, not structure. $\lambda=5000$ is too strong: it irons the broad 3 mHz
# bump flat, which *is* structure. The middle value is roughly right, and the fact
# that "roughly right" has to be chosen is the whole difficulty — in a real
# analysis there is no truth curve to check against, so $\lambda$ is either
# marginalised over or set by cross-validation.
