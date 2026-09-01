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
# # Part 1: Bayesian inference from first principles
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
# Build Bayesian parameter estimation from a signal and noise model, without hiding any step behind Bilby.
#
# :::{admonition} Live route
# :class: tip
#
# Follow Sections 1--5 in order: model, prior, likelihood, posterior checks, then the gravitational-wave noise model. Stop at the clear end-of-live-route marker.
# :::

# %% [markdown]
# ## The four pieces of Bayes' theorem
#
# Bayesian inference updates uncertainty. It combines what the model allowed
# before seeing these data with how well each allowed parameter value explains
# the data:
#
# $$
# p(\theta\mid d,M)=
# \frac{\mathcal L(d\mid\theta,M)\,\pi(\theta\mid M)}{\mathcal Z}.
# $$
#
# | Piece | Plain-language question |
# | --- | --- |
# | parameter $\theta$ | what unknown quantity are we trying to learn? |
# | prior $\pi(\theta)$ | what values did the model allow before these data? |
# | likelihood $\mathcal L(d\mid\theta)$ | how well would each proposed value explain the observed data? |
# | posterior $p(\theta\mid d)$ | what values remain plausible after combining prior and likelihood? |
# | evidence $\mathcal Z$ | what normalises the posterior, and how well did the complete model predict the data? |
#
# For parameter estimation, the reusable calculation is
#
# $$
# \text{posterior}\propto\text{likelihood}\times\text{prior}.
# $$
#
# We will calculate every part on a two-parameter grid before introducing any
# sampling algorithm. A posterior is a distribution, not just the location of
# the best-fitting line.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
rng = np.random.default_rng(20260817)
plt.style.use("seaborn-v0_8-whitegrid")


def show_animation(animation):
    # H.264 video: ~40x smaller in the notebook than one PNG per frame
    try:
        return display(HTML(animation.to_html5_video()))
    except RuntimeError:  # ffmpeg unavailable: fall back to per-frame PNGs
        return display(HTML(animation.to_jshtml()))


print("Running in Colab:", IN_COLAB)

# %% [markdown]
# ## 1. Data and a signal model
#
# Assume $d_i=m t_i+c+n_i$ and independent Gaussian noise $n_i\sim\mathcal N(0,\sigma^2)$. Every likelihood statement is conditional on assumptions like these.

# %% [markdown]
# **Predict before running:** If the same noisy data admit several plausible
# lines, what information is missing from a best-fit line alone? Keep that answer
# in mind when the posterior appears below.

# %%
true_parameters = {"m": 0.5, "c": 0.2}
sigma = 3.0
time = np.linspace(0, 10, 100)


def signal_model(time, m, c):
    return m * time + c


data = signal_model(time, **true_parameters) + rng.normal(0, sigma, time.size)
fig, ax = plt.subplots(figsize=(8, 3.3))
ax.plot(time, data, "o", ms=3, label="data")
ax.plot(time, signal_model(time, **true_parameters), lw=2, label="injected signal")
ax.set(xlabel="time", ylabel="observation", title="Data = signal + noise")
ax.legend()
plt.show()

# %% [markdown]
# ## 2. Priors and prior predictive checks
#
# Take $m\sim\mathrm{Uniform}(0,1.5)$ and $c\sim\mathrm{Uniform}(-5,5)$. A prior is part of the model, not an afterthought. Drawing curves from it checks whether the model can plausibly describe the data before inference.

# %% [markdown]
# **Predict before running:** Which is the more serious warning sign: a prior
# that is broad, or a prior predictive curve that cannot resemble the observed
# data? Explain why before looking at the draw.

# %%
n_prior = 2500
prior_m = rng.uniform(0, 1.5, n_prior)
prior_c = rng.uniform(-5, 5, n_prior)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].hist(prior_m, bins=30, density=True, histtype="step", label="m")
axes[0].hist(prior_c, bins=30, density=True, histtype="step", label="c")
axes[0].set(xlabel="parameter value", ylabel="prior density", title="Marginal priors")
axes[0].legend()
axes[1].plot(time, data, "o", ms=3, color="k")
for m, c in zip(prior_m[:40], prior_c[:40]):
    axes[1].plot(time, signal_model(time, m, c), color="C0", alpha=0.08)
axes[1].set(xlabel="time", ylabel="observation", title="Prior predictive curves")
plt.show()


# %% [markdown]
# ## 3. Gaussian likelihood
#
# $$
# \log\mathcal L(d\mid m,c)=-\frac12\sum_i\left[
# \frac{(d_i-mt_i-c)^2}{\sigma^2}+\log(2\pi\sigma^2)\right].
# $$
#
# Changing the assumed noise scale changes the width of the posterior. If the noise model is wrong, a mathematically correct sampler still gives a misleading answer.

# %% [markdown]
# ### Code studio: write the Gaussian log likelihood
#
# Take 5 minutes in pairs. Translate the equation above into NumPy.
#
# - Compute the residual with `signal_model`.
# - Return one scalar log likelihood.
# - Keep the normalisation term: it matters when noise models are compared.
# - Run the self-check. A correct function reports `check passed`.
#
# The cell is deliberately safe to run before you fill it in.

# %%
def student_log_likelihood(m, c):
    # YOUR CODE HERE
    return None


student_value = student_log_likelihood(true_parameters["m"], true_parameters["c"])
if student_value is None:
    print("Your turn: replace the placeholder in student_log_likelihood.")
else:
    residual = data - signal_model(time, true_parameters["m"], true_parameters["c"])
    expected = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma**2))
    np.testing.assert_allclose(student_value, expected)
    print("check passed")


# %% [markdown]
# <details>
# <summary>Show one possible solution</summary>
#
# ```python
# def student_log_likelihood(m, c):
#     residual = data - signal_model(time, m, c)
#     return -0.5 * np.sum(
#         (residual / sigma) ** 2 + np.log(2 * np.pi * sigma**2)
#     )
# ```
#
# </details>

# %% fqcp_figure="basics-grid-posterior"
def log_likelihood(m, c):
    residual = data - signal_model(time, m, c)
    return -0.5 * np.sum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma**2))


m_grid = np.linspace(0, 1.5, 141)
c_grid = np.linspace(-5, 5, 161)
M, C = np.meshgrid(m_grid, c_grid, indexing="ij")
logL = np.array([[log_likelihood(m, c) for c in c_grid] for m in m_grid])
log_prior = np.zeros_like(logL)  # constant inside this finite grid
log_posterior = logL + log_prior
posterior = np.exp(log_posterior - log_posterior.max())
posterior /= np.trapezoid(np.trapezoid(posterior, c_grid, axis=1), m_grid)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True, sharey=True)
for ax, values, title in zip(
    axes,
    [np.exp(log_prior), np.exp(logL - logL.max()), posterior],
    ["prior", "likelihood", "posterior"],
):
    image = ax.contourf(m_grid, c_grid, values.T, levels=24, cmap="magma")
    ax.plot(true_parameters["m"], true_parameters["c"], "c*", ms=10)
    ax.set(title=title, xlabel="slope m")
axes[0].set_ylabel("intercept c")
plt.show()

# %% [markdown]
# <details>
# <summary><i>Expected output &mdash; open this if your cell has not run yet</i></summary>
#
# <img src="https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/assets/expected/basics-grid-posterior.png" alt="expected output: basics-grid-posterior" style="max-width:100%">
#
# </details>

# %% [markdown]
# The posterior is a ridge: increasing the slope can be compensated by decreasing the intercept. Marginalisation integrates over the other parameter; it is not the same as holding it at a best-fit value.

# %%
p_m = np.trapezoid(posterior, c_grid, axis=1)
p_c = np.trapezoid(posterior, m_grid, axis=0)


def interval(grid, density):
    cdf = np.r_[0, np.cumsum((density[:-1] + density[1:]) * np.diff(grid) / 2)]
    cdf /= cdf[-1]
    return np.interp([0.05, 0.5, 0.95], cdf, grid)


fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, grid, density, name, truth in zip(
    axes, [m_grid, c_grid], [p_m, p_c], ["m", "c"], true_parameters.values()
):
    q = interval(grid, density)
    ax.plot(grid, density)
    ax.axvline(truth, color="k", ls="--")
    ax.axvspan(q[0], q[2], alpha=0.2)
    ax.set(
        xlabel=name,
        ylabel="marginal posterior",
        title=f"median {q[1]:.2f}; 90% [{q[0]:.2f}, {q[2]:.2f}]",
    )
plt.show()

# %% [markdown]
# :::{admonition} A correct sampler cannot repair a wrong likelihood
# :class: warning
#
# Keep the same data but tell the likelihood that the noise standard deviation is
# half its true value. The posterior becomes narrower because the calculation
# thinks the data are more informative—not because it learned more. This is why
# we check residuals and the noise/PSD model, rather than treating a sharp
# posterior as success.
# :::

# %%
wrong_sigma = sigma / 2
wrong_logL = np.array(
    [
        [
            -0.5 * np.sum(((data - signal_model(time, m, c)) / wrong_sigma) ** 2)
            for c in c_grid
        ]
        for m in m_grid
    ]
)
wrong_posterior = np.exp(wrong_logL - wrong_logL.max())
wrong_posterior /= np.trapezoid(np.trapezoid(wrong_posterior, c_grid, axis=1), m_grid)
wrong_p_m = np.trapezoid(wrong_posterior, c_grid, axis=1)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(m_grid, p_m, label="correct noise model")
ax.plot(m_grid, wrong_p_m, label="assumed noise is too small")
ax.axvline(true_parameters["m"], color="k", ls="--", label="injected slope")
ax.set(
    xlabel="slope m",
    ylabel="marginal posterior density",
    title="Wrong noise model: overconfident, not more informed",
)
ax.legend()
plt.show()


# %% [markdown]
# ### What does the evidence do?
#
# For a model $M$, the evidence averages the likelihood over its **normalised**
# prior,
#
# $$
# \mathcal Z_M=\int \mathcal L(d\mid\theta,M)\,\pi(\theta\mid M)\,d\theta.
# $$
#
# The next cell compares a line with free slope and intercept ($M_1$) against a
# line forced through zero ($M_0$). A model does not win merely because its best
# fit is higher: extra prior volume that fits poorly reduces its evidence. This
# is the Bayesian form of an Occam penalty, and it also means Bayes factors must
# be reported with their priors.

# %% [markdown]
# **Predict before running:** If we widen a prior in a direction that the data
# do not constrain, should the posterior peak move, the evidence move, both, or
# neither?

# %%
def log_trapezoid_exp(log_values, grid, axis=-1):
    """Stable log of the trapezoidal integral of exp(log_values)."""
    reference = np.max(log_values)
    integral = np.trapezoid(np.exp(log_values - reference), grid, axis=axis)
    return reference + np.log(integral)


# M1: m and c are both free with a uniform prior on the plotted rectangle.
log_z_free_intercept = log_trapezoid_exp(
    np.array([log_trapezoid_exp(logL[row], c_grid) for row in range(len(m_grid))]),
    m_grid,
) - np.log((m_grid[-1] - m_grid[0]) * (c_grid[-1] - c_grid[0]))

# M0: c=0 exactly and only m is free.
logL_zero_intercept = np.array([log_likelihood(m, 0.0) for m in m_grid])
log_z_zero_intercept = log_trapezoid_exp(logL_zero_intercept, m_grid) - np.log(
    m_grid[-1] - m_grid[0]
)

log_bayes_factor = log_z_free_intercept - log_z_zero_intercept
print(f"log Z (free intercept): {log_z_free_intercept:.2f}")
print(f"log Z (zero intercept): {log_z_zero_intercept:.2f}")
print(f"log Bayes factor, free/zero: {log_bayes_factor:.2f}")

# %% [markdown]
# ### Fast animation: information accumulates
#
# Each frame uses a longer prefix of the same dataset. The posterior does not have to shrink monotonically for every noise realisation, but its typical scale contracts as information accumulates.

# %%
model_cube = M[:, :, None] * time[None, None, :] + C[:, :, None]
cumulative_sse = np.cumsum((data[None, None, :] - model_cube) ** 2, axis=2)
frame_sizes = np.arange(8, time.size + 1, 6)
slope_densities = []
for n_used in frame_sizes:
    frame_logp = -0.5 * cumulative_sse[:, :, n_used - 1] / sigma**2
    frame_p = np.exp(frame_logp - frame_logp.max())
    marginal = np.trapezoid(frame_p, c_grid, axis=1)
    slope_densities.append(marginal / np.trapezoid(marginal, m_grid))
fig, ax = plt.subplots(figsize=(7, 3.2))
(line,) = ax.plot([], [], color="C3")
ax.axvline(true_parameters["m"], color="k", ls="--")
ax.set(
    xlim=(m_grid.min(), m_grid.max()),
    ylim=(0, 1.1 * np.max(slope_densities)),
    xlabel="slope m",
    ylabel="posterior density",
)


def animate_learning(i):
    line.set_data(m_grid, slope_densities[i])
    ax.set_title(f"posterior after {frame_sizes[i]} observations")
    return (line,)


learning_animation = FuncAnimation(
    fig, animate_learning, frames=len(frame_sizes), interval=150
)
plt.close(fig)
show_animation(learning_animation)


# %% [markdown]
# ## 4. Posterior predictive check
#
# Draw parameter pairs from the posterior, map each through the signal model, and
# **add a noise draw**. That is the posterior predictive distribution: it predicts
# *data*, not the noise-free curve, so it is the only version the observed points
# can actually be compared against.
#
# A summary statistic turns the picture into a number. Take
# $\chi^2=\sum_i\left[(d_i-h_i(\theta))/\sigma\right]^2$: the posterior predictive
# *p*-value is the fraction of replicated datasets whose $\chi^2$ is at least as
# large as the observed one. Values near 0 or 1 mean the model cannot produce data
# like ours. It is a falsification test, not a score to maximise.

# %% fqcp_figure="basics-posterior-predictive"
def posterior_predictive(model, first, second, weights, observed, n_draw=500):
    """Replicate datasets from the posterior and score them against `observed`."""
    flat = (weights / weights.sum()).ravel()
    draw = rng.choice(flat.size, size=n_draw, replace=True, p=flat)
    curves = np.array(
        [model(time, a, b) for a, b in zip(first.ravel()[draw], second.ravel()[draw])]
    )
    replicas = curves + rng.normal(0, sigma, curves.shape)  # predict data, not curves
    chi2 = lambda residual: ((residual / sigma) ** 2).sum(axis=1)
    p_value = (chi2(replicas - curves) >= chi2(observed - curves)).mean()
    return curves, replicas, p_value


curves, replicas, p_value = posterior_predictive(signal_model, M, C, posterior, data)

fig, (ax, rx) = plt.subplots(1, 2, figsize=(11, 3.3))
ax.fill_between(
    time,
    *np.quantile(replicas, [0.05, 0.95], axis=0),
    alpha=0.2,
    color="C0",
    label="90% predictive data band",
)
ax.fill_between(
    time,
    *np.quantile(curves, [0.05, 0.95], axis=0),
    alpha=0.45,
    color="C1",
    label="90% signal band",
)
ax.plot(time, data, "o", ms=3, color="k", label="data")
ax.set(
    xlabel="time",
    ylabel="observation",
    title=f"Posterior predictive, p = {p_value:.2f}",
)
ax.legend(fontsize=8)

rx.axhline(0, color="k", lw=1)
rx.fill_between(
    time,
    *np.quantile(replicas - curves, [0.05, 0.95], axis=0),
    alpha=0.25,
    color="C0",
    label="90% predictive residual band",
)
rx.plot(time, np.median(data - curves, axis=0), color="C3", label="observed residual")
rx.set(xlabel="time", ylabel="data - signal", title="No structure left over")
rx.legend(fontsize=8)
plt.show()


# %% [markdown]
# <details>
# <summary><i>Expected output &mdash; open this if your cell has not run yet</i></summary>
#
# <img src="https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/assets/expected/basics-posterior-predictive.png" alt="expected output: basics-posterior-predictive" style="max-width:100%">
#
# </details>

# %% [markdown]
# ### When the model is wrong
#
# Every check so far has passed, so none of them has yet been shown to *do*
# anything. Here the data are generated by an exponential,
# $h(t)=A\,(e^{t/\tau}-1)$, and fitted with both the exponential model and the
# straight line. The line is simply the wrong shape.
#
# Look at the left column before reading the numbers. The wrong model does not
# look absurd: the posterior is tight, the fit passes through the data, and most
# points sit inside the band. **A posterior cannot tell you its model is wrong** —
# it only reports the best available parameters of whatever model it was given.
#
# The two checks that do notice are the residual panel, where the misfit appears
# as a slow arc instead of scatter, and the *p*-value.

# %% fqcp_figure="basics-wrong-model"
def exponential_model(time, amplitude, tau):
    return amplitude * (np.exp(time / tau) - 1.0)


true_curved = {"amplitude": 1.0, "tau": 3.0}
curved_data = exponential_model(time, **true_curved) + rng.normal(0, sigma, time.size)


def fit_on_grid(model, observed, first_grid, second_grid):
    """Grid posterior and evidence for a two-parameter model with a flat prior."""
    first, second = np.meshgrid(first_grid, second_grid, indexing="ij")
    log_l = np.array(
        [
            [
                -0.5
                * np.sum(
                    ((observed - model(time, a, b)) / sigma) ** 2
                    + np.log(2 * np.pi * sigma**2)
                )
                for b in second_grid
            ]
            for a in first_grid
        ]
    )
    log_z = log_trapezoid_exp(
        np.array([log_trapezoid_exp(row, second_grid) for row in log_l]), first_grid
    ) - np.log((first_grid[-1] - first_grid[0]) * (second_grid[-1] - second_grid[0]))
    return first, second, np.exp(log_l - log_l.max()), log_z


candidates = {
    "linear (wrong)": (
        signal_model,
        np.linspace(-2, 6, 141),
        np.linspace(-15, 15, 161),
    ),
    "exponential (true)": (
        exponential_model,
        np.linspace(0, 4, 141),
        np.linspace(1, 10, 161),
    ),
}

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
evidences, p_values = {}, {}
for column, (name, (model, first_grid, second_grid)) in zip(axes.T, candidates.items()):
    first, second, weights, log_z = fit_on_grid(
        model, curved_data, first_grid, second_grid
    )
    curves, replicas, p_value = posterior_predictive(
        model, first, second, weights, curved_data
    )
    evidences[name] = log_z
    p_values[name] = p_value

    top, bottom = column
    top.fill_between(
        time, *np.quantile(replicas, [0.05, 0.95], axis=0), alpha=0.2, color="C0"
    )
    top.fill_between(
        time, *np.quantile(curves, [0.05, 0.95], axis=0), alpha=0.5, color="C1"
    )
    top.plot(time, curved_data, "o", ms=3, color="k")
    top.set_title(f"{name}\nlog Z = {log_z:.1f},   p = {p_value:.2f}")

    bottom.axhline(0, color="k", lw=1)
    bottom.fill_between(
        time,
        *np.quantile(replicas - curves, [0.05, 0.95], axis=0),
        alpha=0.25,
        color="C0",
    )
    bottom.plot(time, np.median(curved_data - curves, axis=0), color="C3")
    bottom.set(xlabel="time", ylim=(-12, 12))

axes[0, 0].set_ylabel("observation")
axes[1, 0].set_ylabel("data - signal")
plt.show()

for name, p_value in p_values.items():
    print(f"{name:20s} posterior predictive p = {p_value:.3f}")
log_bf = evidences["exponential (true)"] - evidences["linear (wrong)"]
print(f"log Bayes factor, exponential over linear: {log_bf:.1f}")
print(f"the two prior volumes differ by only {np.log(240 / 36):.1f} in log evidence")

# %% [markdown]
# <details>
# <summary><i>Expected output &mdash; open this if your cell has not run yet</i></summary>
#
# <img src="https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/assets/expected/basics-wrong-model.png" alt="expected output: basics-wrong-model" style="max-width:100%">
#
# </details>

# %% [markdown]
# The line is not merely worse, it is excluded. Its *p*-value is zero to three
# decimal places: not one replicated dataset drawn from its own posterior is as
# badly fitted as the real data. The evidence agrees independently, and by a wide
# margin.
#
# Two cautions that generalise to the gravitational-wave chapters:
#
# - **The Bayes factor is not free of the priors.** These two models were given
#   different prior boxes, worth about $1.9$ in log evidence. That is negligible
#   against the log Bayes factor printed above, but it would not be negligible
#   against a log Bayes factor of $3$.
# - **A passing check is not proof.** The exponential model passes here because it
#   is the model that generated the data. In a real analysis the true shape is
#   never on the menu, and a *p*-value that is merely non-extreme means only that
#   this particular statistic found nothing wrong. Note also that these *p*-values
#   are not uniformly distributed under the true model: the data were used both to
#   fit and to test, which makes them conservative.
#
# This is the whole reason gravitational-wave analyses carry residual tests and
# waveform-systematics studies alongside their posteriors.

# %% [markdown]
# ### Question
#
# Change the assumed noise standard deviation from `sigma` to `sigma / 2`. Recompute a normalized slope posterior and report how its 90% interval changes. Why is a narrower posterior not automatically a better result?
#
# Write your answer in the cell immediately below. The starter runs safely before
# you edit it, so the complete notebook remains reproducible.

# %%
# Your code here
assumed_sigma = sigma / 2
print("Exercise ready:", "recompute the posterior with assumed_sigma")

# %% [markdown]
# <details>
# <summary>Hint</summary>
#
# Reuse the `M`, `C`, `data`, and `time` arrays from the grid calculation. Only the likelihood width changes.
# </details>
#
# <details>
# <summary>Solution and check</summary>
#
# ```python
# residual = data[None, None, :] - (M[:, :, None] * time + C[:, :, None])
# log_like = -0.5 * np.sum((residual / assumed_sigma) ** 2, axis=-1)
# p = np.exp(log_like - np.max(log_like))
# p /= np.trapezoid(np.trapezoid(p, c_grid, axis=1), m_grid)
# p_m_test = np.trapezoid(p, c_grid, axis=1)
# print(interval(m_grid, p_m_test))
# # The interval contracts because the likelihood was made overconfident, not
# # because the data became more informative.
# ```
# </details>

# %% [markdown]
# ## 5. The gravitational-wave bridge: PSD and Whittle likelihood
#
# A power spectral density (PSD) describes how a stationary random process's
# variance is distributed over frequency. For one-sided $S_n(f)$,
# $S_n(f)\,df$ is the expected noise variance in a small positive-frequency band.
# Its units are strain$^2$/Hz; the amplitude spectral density (ASD)
# $\sqrt{S_n(f)}$ has units strain/$\sqrt{\mathrm{Hz}}$.
#
# For an approximately stationary, Gaussian time series, well-behaved Fourier
# coefficients are approximately independent complex Gaussians. This gives the
# Whittle approximation
#
# $$
# \log \mathcal L(d\mid\theta,S_n)
# \simeq -\frac{1}{2}\sum_k
# \left[\frac{4\,\Delta f\,|\tilde d_k-\tilde h_k(\theta)|^2}{S_n(f_k)}
# +\log S_n(f_k)\right]+C.
# $$
#
# When the PSD is fixed, the $\log S_n$ term is constant and we often write
#
# $$
# \log\mathcal L=-\frac12(d-h\mid d-h)+C,\qquad
# (a\mid b)=4\,\mathrm{Re}\sum_k
# \frac{\tilde a_k\tilde b_k^*}{S_n(f_k)}\Delta f.
# $$
#
# The inverse PSD is therefore a frequency-dependent weight: residual power in a
# quiet band matters more. Gaps, strong lines, spectral leakage, and
# non-stationarity couple Fourier bins and weaken the simple independence
# approximation.

# %%
from scipy.signal import welch

sample_rate = 512
duration = 32
noise_time = np.arange(0, duration, 1 / sample_rate)
noise_frequency = np.fft.rfftfreq(noise_time.size, 1 / sample_rate)

# A deliberately non-white spectrum: large low-frequency noise and a mild
# high-frequency rise. The absolute normalisation is arbitrary in this toy.
noise_shape = (
    1 + (30 / np.maximum(noise_frequency, 1)) ** 4 + (noise_frequency / 180) ** 2
)
white_draw = rng.normal(size=noise_time.size)
coloured_noise = np.fft.irfft(
    np.fft.rfft(white_draw) * np.sqrt(noise_shape), n=noise_time.size
)
psd_frequency, estimated_psd = welch(
    coloured_noise,
    fs=sample_rate,
    nperseg=2048,
    average="median",
)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
axes[0].plot(noise_time[: 4 * sample_rate], coloured_noise[: 4 * sample_rate])
axes[0].set(
    xlabel="time [s]",
    ylabel="noise [toy units]",
    title="One coloured-noise realisation",
)
axes[1].loglog(psd_frequency[1:], np.sqrt(estimated_psd[1:]))
axes[1].set(
    xlabel="frequency [Hz]",
    ylabel=r"ASD [toy units/$\sqrt{\mathrm{Hz}}$]",
    title="Welch estimate of the ASD",
)
plt.show()

# %% [markdown]
# ### Optional audio analogy: hear and see what whitening does
#
# - This is **not detector strain converted to sound**. It is an audible toy: a
#   chirp buried in coloured noise.
# - Whitening divides each Fourier component by the noise ASD,
#   $\tilde d(f)\rightarrow\tilde d(f)/\sqrt{S_n(f)}$, which is the same
#   inverse-noise weighting the Whittle likelihood applies.
# - Listen first, then look at the spectrograms. The low-frequency noise wall
#   carries almost all the power, which is why the raw clip sounds like rumble.
# - After whitening every frequency carries comparable noise, so the chirp
#   becomes the loudest and the brightest feature.

# %%
from IPython.display import Audio
from scipy.signal import chirp

audio_rate = 4096
audio_duration = 3.0
audio_time = np.arange(0, audio_duration, 1 / audio_rate)
audio_frequency = np.fft.rfftfreq(audio_time.size, 1 / audio_rate)

audio_noise_shape = 1 + (450 / np.maximum(audio_frequency, 20)) ** 4
audio_noise = np.fft.irfft(
    np.fft.rfft(rng.normal(size=audio_time.size)) * np.sqrt(audio_noise_shape),
    n=audio_time.size,
)
audio_signal = 0.8 * chirp(
    audio_time,
    f0=250,
    f1=1200,
    t1=audio_duration,
    method="quadratic",
)
audio_data = audio_noise + audio_signal

whitened_audio = np.fft.irfft(
    np.fft.rfft(audio_data) / np.sqrt(audio_noise_shape), n=audio_time.size
)


def safe_audio(values):
    values = values / np.max(np.abs(values))
    return Audio(values, rate=audio_rate, normalize=False)


print("Coloured data: the chirp is partly masked")
display(safe_audio(audio_data))
print("Whitened data: frequencies are placed on a comparable noise scale")
display(safe_audio(whitened_audio))

# %%
from scipy.signal import spectrogram

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 3.9),
    sharey=True,
    gridspec_kw={"width_ratios": [1, 3, 3], "wspace": 0.08},
)
f_top = 1600

axes[0].loglog(np.sqrt(audio_noise_shape[1:]), audio_frequency[1:], color="C3")
axes[0].invert_xaxis()
axes[0].set(
    ylim=(20, f_top),
    xlabel="noise ASD",
    ylabel="frequency [Hz]",
    title="the weight",
)

for ax, series, title in zip(
    axes[1:], [audio_data, whitened_audio], ["coloured data", "whitened data"]
):
    spec_f, spec_t, power = spectrogram(
        series, fs=audio_rate, nperseg=256, noverlap=224
    )
    band = (spec_f > 20) & (spec_f < f_top)
    decibels = 10 * np.log10(power[band] + 1e-20)
    ax.pcolormesh(
        spec_t,
        spec_f[band],
        decibels,
        shading="auto",
        cmap="magma",
        vmin=np.percentile(decibels, 5),
        vmax=np.percentile(decibels, 99.7),
    )
    ax.set(xlabel="time [s]", title=title)
axes[1].set_yscale("log")
plt.show()

# %% [markdown]
# :::{admonition} End of the live route
# :class: important
#
# You now have the complete inference chain used later in the course:
#
# $$
# \text{data} + \text{signal model} + \text{noise model}
# \longrightarrow \text{likelihood}
# \longrightarrow \text{posterior}
# \longrightarrow \text{checks}.
# $$
#
# The remaining sections explain how real analyses explore a posterior when a
# grid is impossible. They are reference material for a second pass.
# :::

# %% [markdown]
# ## Read later: replacing the grid with samplers
#
# The live lesson used a grid because it makes the prior, likelihood, and
# posterior visible. Real gravitational-wave problems have too many parameters
# for a grid. The next two sections retain readable teaching implementations of
# the two main alternatives: Metropolis--Hastings for posterior samples and
# nested sampling for posterior samples plus evidence.

# %% [markdown]
# ## Sampler 1: Metropolis--Hastings
#
# - A grid with $n$ points per axis and $D$ parameters costs $n^D$ likelihood
#   evaluations.
#
# $$
# \text{cost}=n^D,\qquad D_{\rm BBH}\approx15,\ n=20
# \ \Rightarrow\ 20^{15}\approx3\times10^{19}.
# $$
#
# - At 1 ms per waveform that is about a **billion years**.
# - The posterior occupies a vanishingly small fraction of that volume, so almost
#   every grid point is wasted.
# - Stochastic samplers spend their effort where the posterior has mass. The next
#   three sections build the two that dominate gravitational-wave work:
#   **MCMC** for parameters, **nested sampling** for evidence.

# %%
dimensions = np.arange(1, 16)
grid_cost = 20.0**dimensions
seconds_per_likelihood = 1e-3

fig, ax = plt.subplots(figsize=(7.5, 3.3))
ax.semilogy(dimensions, grid_cost * seconds_per_likelihood / (3600 * 24 * 365), "o-")
ax.axhline(1, color="C3", ls="--", label="one year of computing")
ax.set(
    xlabel="number of parameters",
    ylabel="grid cost [years]",
    title="A 20-point-per-axis grid at 1 ms per likelihood",
)
ax.legend()
plt.show()

print(f"Two parameters: {20**2:,} evaluations")
print(f"Fifteen parameters: {20**15:.2e} evaluations")

# %% [markdown]
# ### Metropolis-Hastings in twelve lines
#
# From the current point $\theta$, repeat:
#
# $$
# \theta'=\theta+\mathcal N(0,\Sigma_{\rm prop}),\qquad
# \alpha=\min\left[1,\;
# \frac{\mathcal L(\theta')\,\pi(\theta')}{\mathcal L(\theta)\,\pi(\theta)}\right],
# $$
#
# accept $\theta'$ with probability $\alpha$, otherwise **store $\theta$ again**.
#
# - Only the *ratio* is needed, so the evidence $\mathcal Z$ cancels. This is why
#   MCMC works when the normalisation is unknown.
# - Repeating a rejected point is not a bug: it is how the chain accumulates
#   density where the posterior is large.
# - The output is a set of **correlated** draws whose histogram converges to the
#   posterior. Correlated is fine; it just costs effective samples (Section 5.3).
# - Uphill moves are always accepted; downhill moves are accepted sometimes. That
#   is what stops the chain collapsing onto the maximum-likelihood point.

# %%
PRIOR_BOX = np.array([[0.0, 1.5], [-5.0, 5.0]])  # rows: m, c


def log_posterior(theta):
    """Unnormalised log posterior: flat prior inside the box, zero outside."""
    if np.any(theta < PRIOR_BOX[:, 0]) or np.any(theta > PRIOR_BOX[:, 1]):
        return -np.inf
    return log_likelihood(theta[0], theta[1])


def metropolis(log_target, start, n_steps, step_size, rng):
    """Random-walk Metropolis. Returns the chain and the acceptance fraction."""
    chain = np.empty((n_steps, len(start)))
    current = np.asarray(start, dtype=float)
    current_logp = log_target(current)
    n_accepted = 0
    for step in range(n_steps):
        proposal = current + rng.normal(0.0, step_size)
        proposal_logp = log_target(proposal)
        if np.log(rng.uniform()) < proposal_logp - current_logp:
            current, current_logp = proposal, proposal_logp
            n_accepted += 1
        chain[step] = current
    return chain, n_accepted / n_steps


sampler_rng = np.random.default_rng(4)
chain, acceptance = metropolis(
    log_posterior,
    start=[1.35, -4.0],  # deliberately a bad starting guess
    n_steps=6000,
    step_size=np.array([0.12, 0.7]),
    rng=sampler_rng,
)
print(f"acceptance fraction: {acceptance:.2f}")
print(f"chain shape: {chain.shape}")

# %% [markdown]
# ### Animation: watch the chain find the posterior
#
# Two distinct phases to look for:
#
# - **Burn-in** — a directed climb from the deliberately bad start in the corner,
#   where the posterior is negligible. These samples describe where we started,
#   not the posterior, so they are discarded.
# - **Sampling** — the walker wanders up and down the degeneracy ridge. This is
#   the part that is actually a draw from the posterior.
#
# The trace panel on the right is the standard way to spot the transition.

# %%
frame_steps = np.arange(20, 1400, 26)
fig, (walk_ax, trace_ax) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=72)
walk_ax.contour(m_grid, c_grid, posterior.T, levels=6, cmap="magma")
(path,) = walk_ax.plot([], [], lw=0.7, color="C0", alpha=0.8)
(head,) = walk_ax.plot([], [], "o", color="C3", ms=7)
walk_ax.plot(true_parameters["m"], true_parameters["c"], "c*", ms=12)
walk_ax.set(xlim=(0, 1.5), ylim=(-5, 5), xlabel="slope m", ylabel="intercept c")
(trace_line,) = trace_ax.plot([], [], lw=0.8, color="C0")
trace_ax.axhline(true_parameters["m"], color="k", ls="--")
trace_ax.set(xlim=(0, frame_steps[-1]), ylim=(0, 1.5), xlabel="step", ylabel="slope m")


def animate_chain(i):
    n = frame_steps[i]
    path.set_data(chain[:n, 0], chain[:n, 1])
    head.set_data([chain[n - 1, 0]], [chain[n - 1, 1]])
    trace_line.set_data(np.arange(n), chain[:n, 0])
    walk_ax.set_title(f"step {n}")
    return path, head, trace_line


chain_animation = FuncAnimation(
    fig, animate_chain, frames=len(frame_steps), interval=80
)
plt.close(fig)
show_animation(chain_animation)

# %% [markdown]
# ### The proposal scale controls everything
#
# A chain can be correct in principle and useless in practice:
#
# - **Steps too small** — almost everything is accepted, but the walker crawls
#   and never crosses the posterior. Watch the printed mean below: it is badly
#   wrong, from code that has no bug.
# - **Steps too large** — almost every proposal lands somewhere absurd and is
#   rejected, so the chain sits still.
# - **Well tuned** — acceptance around 0.2-0.3 for random-walk Metropolis.
#
# Both failures look the same in the end: a chain that has not forgotten where it
# started. `emcee`, `dynesty`, and `bilby` automate the tuning, but these failure
# modes are exactly what their convergence diagnostics are looking for.

# %%
settings = [
    ("too small", np.array([0.004, 0.02])),
    ("well tuned", np.array([0.12, 0.7])),
    ("too large", np.array([1.2, 7.0])),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
for ax, (label, step_size) in zip(axes, settings):
    trial_chain, trial_acceptance = metropolis(
        log_posterior, [1.35, -4.0], 6000, step_size, np.random.default_rng(4)
    )
    ax.plot(trial_chain[:, 0], lw=0.6)
    ax.axhline(true_parameters["m"], color="k", ls="--")
    ax.set(
        xlabel="step",
        title=f"{label}\nacceptance {trial_acceptance:.2f}",
    )
    print(
        f"{label:>10}: acceptance {trial_acceptance:.2f}, "
        f"posterior mean m = {trial_chain[500:, 0].mean():.3f}"
    )
axes[0].set_ylabel("slope m")
axes[0].set_ylim(0, 1.5)
plt.show()

# %% [markdown]
# ### Diagnostics: burn-in and effective sample size
#
# Consecutive samples are correlated, so $N$ stored samples are worth fewer than
# $N$ independent draws. With autocorrelation $\rho(k)$ at lag $k$,
#
# $$
# N_{\rm eff}\simeq\frac{N}{1+2\sum_{k\ge1}\rho(k)},\qquad
# \text{Monte Carlo error}\propto\frac{1}{\sqrt{N_{\rm eff}}}.
# $$
#
# - $N_{\rm eff}$, **not** the raw chain length, sets the error on every
#   posterior summary you quote.
# - A million highly correlated samples can carry less information than a
#   thousand independent ones.
# - Thinning (keeping every $k$-th sample) saves storage. It does not improve
#   $N_{\rm eff}$, so it never buys accuracy.
# - Rule of thumb: report a number only if $N_{\rm eff}$ is in the hundreds.

# %%
burn_in = 500
samples = chain[burn_in:]


def autocorrelation(x):
    """Normalised autocorrelation function of a 1D chain."""
    x = x - x.mean()
    acf = np.correlate(x, x, mode="full")[x.size - 1 :]
    return acf / acf[0]


def effective_sample_size(x):
    acf = autocorrelation(x)
    first_small = np.argmax(acf < 0.05)
    cutoff = acf.size if first_small == 0 else first_small
    return x.size / (1 + 2 * acf[1:cutoff].sum())


fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
axes[0].plot(chain[:, 0], lw=0.6)
axes[0].axvspan(0, burn_in, color="C3", alpha=0.2, label="discarded burn-in")
axes[0].axhline(true_parameters["m"], color="k", ls="--")
axes[0].set(xlabel="step", ylabel="slope m", title="Trace")
axes[0].legend()
for index, name in enumerate(["m", "c"]):
    axes[1].plot(autocorrelation(samples[:, index])[:200], label=name)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set(xlabel="lag [steps]", ylabel=r"$\rho$", title="Autocorrelation")
axes[1].legend()
plt.show()

for index, name in enumerate(["m", "c"]):
    print(
        f"{name}: N = {samples.shape[0]}, "
        f"N_eff = {effective_sample_size(samples[:, index]):.0f}"
    )

# %% [markdown]
# ### The corner plot, and a check against the grid
#
# A corner plot is the standard way to display a multi-dimensional posterior: 1D
# marginals on the diagonal, 2D marginals below. Because this problem is small
# enough to solve both ways, we can overlay the exact grid marginals in orange.
# Agreement is the check that the sampler is doing its job, and it is the only
# reason to trust the sampler on problems where no grid is possible.

# %% fqcp_figure="basics-corner-check"
import subprocess
import sys

try:
    import corner
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "corner"])
    import corner

corner_figure = corner.corner(
    samples,
    labels=["slope m", "intercept c"],
    truths=[true_parameters["m"], true_parameters["c"]],
    quantiles=[0.05, 0.5, 0.95],
    show_titles=True,
    title_fmt=".3f",
)
corner_axes = np.array(corner_figure.axes).reshape(2, 2)
for axis, grid, marginal in [
    (corner_axes[0, 0], m_grid, p_m),
    (corner_axes[1, 1], c_grid, p_c),
]:
    # corner draws counts, not a density, so rescale the exact curve to match.
    axis.plot(grid, marginal * axis.get_ylim()[1] / marginal.max(), color="C1", lw=2)
corner_figure.suptitle("MCMC samples vs exact grid marginals (orange)", y=1.02)
plt.show()

print(f"grid    : m = {np.trapezoid(p_m * m_grid, m_grid):.4f}")
print(f"sampler : m = {samples[:, 0].mean():.4f}")


# %% [markdown]
# <details>
# <summary><i>Expected output &mdash; open this if your cell has not run yet</i></summary>
#
# <img src="https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/assets/expected/basics-corner-check.png" alt="expected output: basics-corner-check" style="max-width:100%">
#
# </details>

# %% [markdown]
# ## Sampler 2: nested sampling and evidence
#
# MCMC gives parameters but not $\mathcal Z$. Nested sampling gives both, by
# reordering the integral along **prior volume**. Let $X(\lambda)$ be the
# fraction of the prior with $\mathcal L>\lambda$. Then a $D$-dimensional
# integral becomes a one-dimensional one:
#
# $$
# \mathcal Z=\int\mathcal L(\theta)\,\pi(\theta)\,d\theta
# =\int_0^1\mathcal L(X)\,dX
# \;\simeq\;\sum_i\mathcal L_i\,\Delta X_i .
# $$
#
# The algorithm:
#
# 1. Draw $N_{\rm live}$ points from the prior.
# 2. Delete the **worst** one and record its likelihood.
# 3. Replace it by a new draw from the prior, restricted to
#    $\mathcal L>\mathcal L_{\rm worst}$.
# 4. Repeat. Each deletion shrinks the volume by a known factor,
#    $X_i\approx e^{-i/N_{\rm live}}$.
#
# - The likelihood threshold only ever rises, so the live points contract onto
#   the peak. The animation below shows exactly this.
# - Step 3 is the hard part in real problems, and it is what separates
#   `MultiNest` (ellipsoids) from `PolyChord` and `dynesty` (slice sampling).
#   Ours evolves a copy of a surviving point with a short constrained MCMC.
# - Posterior samples come out free: the deleted points weighted by
#   $\mathcal L_i\Delta X_i$.
# - This is why Bayesian model comparison is practical in GW astronomy at all.

# %%
def nested_sampling(
    log_likelihood_fn, prior_box, n_live=250, n_iterations=1400, n_mcmc=25, rng=None
):
    """A minimal nested sampler with MCMC-based constrained replacement."""
    low, high = prior_box[:, 0], prior_box[:, 1]
    live = rng.uniform(low, high, size=(n_live, low.size))
    live_logl = np.array([log_likelihood_fn(point) for point in live])

    log_evidence = -np.inf
    dead_logl, dead_logw, snapshots = [], [], []

    for iteration in range(n_iterations):
        worst = np.argmin(live_logl)
        log_volume = -iteration / n_live
        log_shell = log_volume + np.log1p(-np.exp(-1.0 / n_live))
        log_evidence = np.logaddexp(log_evidence, live_logl[worst] + log_shell)
        dead_logl.append(live_logl[worst])
        dead_logw.append(log_shell)
        if iteration % 50 == 0:
            snapshots.append((live.copy(), log_volume, log_evidence))

        # Replace the worst point by evolving a copy of a surviving one.
        threshold = live_logl[worst]
        point = live[rng.integers(n_live)].copy()
        point_logl = log_likelihood_fn(point)
        proposal_scale = live.std(axis=0)
        for _ in range(n_mcmc):
            trial = point + rng.normal(0.0, proposal_scale)
            if np.all(trial > low) and np.all(trial < high):
                trial_logl = log_likelihood_fn(trial)
                if trial_logl > threshold:
                    point, point_logl = trial, trial_logl
        live[worst], live_logl[worst] = point, point_logl

    # Add the remaining live points as a final block.
    log_remaining = -n_iterations / n_live - np.log(n_live)
    log_evidence = np.logaddexp(
        log_evidence, np.logaddexp.reduce(live_logl) + log_remaining
    )
    return log_evidence, np.array(dead_logl), np.array(dead_logw), snapshots


log_z_nested, dead_logl, dead_logw, snapshots = nested_sampling(
    lambda point: log_likelihood(point[0], point[1]),
    PRIOR_BOX,
    rng=np.random.default_rng(7),
)

print(f"nested sampling log Z: {log_z_nested:.3f}")
print(f"grid log Z           : {log_z_free_intercept:.3f}")
print(f"difference           : {log_z_nested - log_z_free_intercept:+.3f}")

# %% [markdown]
# ### Animation: the live points contract onto the posterior
#
# Each frame shows the surviving live points. They begin spread over the whole
# prior and are squeezed into the high-likelihood ridge as the likelihood
# threshold rises. The right panel shows the integrand $\mathcal{L}(X)$ against
# $\log X$: the evidence is the area under it, and the visible bump is the
# region of prior volume that actually contributes.

# %%
fig, (live_ax, mass_ax) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=72)
live_ax.contour(m_grid, c_grid, posterior.T, levels=6, cmap="magma")
(live_points,) = live_ax.plot([], [], ".", color="C0", ms=3)
live_ax.set(xlim=(0, 1.5), ylim=(-5, 5), xlabel="slope m", ylabel="intercept c")

log_volume_axis = -np.arange(dead_logl.size) / 250
posterior_mass = np.exp(dead_logl + dead_logw - np.max(dead_logl + dead_logw))
mass_ax.plot(log_volume_axis, posterior_mass, color="0.7")
(mass_line,) = mass_ax.plot([], [], color="C3", lw=2)
mass_ax.set(
    xlabel=r"$\log X$ (log prior volume)",
    ylabel=r"$\mathcal{L}\,\Delta X$ (normalised)",
    title="Where the evidence comes from",
)


def animate_nested(i):
    live, log_volume, running_log_evidence = snapshots[i]
    live_points.set_data(live[:, 0], live[:, 1])
    used = log_volume_axis >= log_volume
    mass_line.set_data(log_volume_axis[used], posterior_mass[used])
    live_ax.set_title(
        f"log X = {log_volume:.1f}, running log Z = {running_log_evidence:.1f}"
    )
    return live_points, mass_line


nested_animation = FuncAnimation(
    fig, animate_nested, frames=len(snapshots), interval=200
)
plt.close(fig)
show_animation(nested_animation)

# %% [markdown]
# ## Further sampler families
#
# You do not need their implementations for this course. The important
# distinctions are:
#
# | Method | Main idea | Main caution |
# | --- | --- | --- |
# | Hamiltonian Monte Carlo / NUTS | gradients move efficiently through a correlated posterior | needs differentiable, well-scaled models |
# | variational inference | optimise an approximate posterior family | can underestimate uncertainty if the family is too simple |
# | simulation-based inference | learn inference from many simulations | must be validated against trusted analyses and simulations |
#
# Return to these methods after you are comfortable identifying the model,
# prior, likelihood, posterior, and checks in a concrete analysis.

# %% [markdown]
# ## Reference: the parameter-estimation checklist
#
# Every analysis in the next two notebooks, and every published gravitational-wave
# result, is built from exactly these pieces.
#
# | Step | Question it answers | Where it can go wrong |
# | --- | --- | --- |
# | signal model $h(\theta)$ | what could have produced the data? | waveform systematics, missing physics |
# | noise model / PSD | what does "a good fit" mean quantitatively? | non-stationarity, lines, glitches, PSD uncertainty |
# | likelihood $\mathcal L(d\mid\theta)$ | how compatible are data and parameters? | wrong noise assumptions, correlated bins |
# | prior $\pi(\theta)$ | what was allowed before these data? | unintended informativeness, hard boundaries |
# | sampler | how do we explore the posterior? | poor tuning, unconverged chains, missed modes |
# | diagnostics | can we trust this particular run? | too few effective samples, no burn-in check |
# | evidence $\mathcal Z$ | which model does the data prefer? | prior-volume dependence, under-converged runs |
# | predictive comparison ($\Delta$elpd) | which model predicts new data better? | selection bias over many models, unreliable $\hat k$ |
# | calibration (P-P) | is the whole pipeline correct? | only detectable over many simulations |
#
# **Vocabulary quick reference**
#
# - *Marginalisation* integrates a nuisance parameter out; *profiling* maximises
#   over it. Section 3 showed these are different, and the LISA notebook measures
#   exactly how different.
# - *Burn-in* is the discarded start of a chain; *thinning* keeps every $k$-th
#   sample. Thinning reduces storage, not Monte Carlo error.
# - *Optimal SNR* assumes a perfect template; *matched-filter SNR* is what you
#   actually recover from data. The LVK notebook computes both.
# - *Credible interval* (Bayesian, probability over parameters) is not a
#   *confidence interval* (frequentist, coverage over repeated experiments).
