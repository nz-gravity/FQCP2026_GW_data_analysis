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
import importlib.util
import os
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
if importlib.util.find_spec("numpyro") is None:
    if IN_COLAB:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "numpyro==0.21.0"]
        )
    else:
        raise ImportError("Install numpyro==0.21.0, or run this notebook in Colab.")

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
# ### Now let a sampler explore the same posterior
#
# The grid above **defined and displayed** the posterior. NumPyro now gives NUTS
# the same prior, signal model, likelihood, and data. The sampler changes how we
# explore the posterior; it does not change the posterior itself.
#
# You do not need to understand NUTS today. Read the model from top to bottom:
# sample a slope, sample an intercept, predict the data, and compare that
# prediction with the observations using the same Gaussian noise model.

# %% fqcp_figure="basics-numpyro-posterior"
import jax.numpy as jnp
from jax import random as jax_random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def numpyro_line_model(time, observed=None):
    m = numpyro.sample("m", dist.Uniform(0.0, 1.5))
    c = numpyro.sample("c", dist.Uniform(-5.0, 5.0))
    mean = m * time + c
    numpyro.sample("data", dist.Normal(mean, sigma), obs=observed)


nuts_run = MCMC(
    NUTS(numpyro_line_model),
    num_warmup=400,
    num_samples=1000,
    progress_bar=False,
)
nuts_run.run(jax_random.PRNGKey(2026), jnp.asarray(time), jnp.asarray(data))
nuts_draws = {name: np.asarray(values) for name, values in nuts_run.get_samples().items()}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].contour(m_grid, c_grid, posterior.T, levels=7, cmap="magma")
axes[0].scatter(nuts_draws["m"], nuts_draws["c"], s=5, alpha=0.15)
axes[0].plot(true_parameters["m"], true_parameters["c"], "c*", ms=11)
axes[0].set(xlabel="slope m", ylabel="intercept c", title="NUTS draws on grid posterior")
for ax, name, grid, exact, truth in zip(
    axes[1:],
    ["m", "c"],
    [m_grid, c_grid],
    [np.trapezoid(posterior, c_grid, axis=1), np.trapezoid(posterior, m_grid, axis=0)],
    true_parameters.values(),
):
    ax.hist(nuts_draws[name], bins=35, density=True, alpha=0.45, label="NUTS")
    ax.plot(grid, exact, color="C3", lw=2, label="exact grid")
    ax.axvline(truth, color="k", ls="--", label="injected")
    ax.set(xlabel=name, ylabel="posterior density", title=f"sampled {name} posterior")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

print("posterior draws:", len(nuts_draws["m"]))

# %% [markdown]
# :::{admonition} Live demonstration, not a convergence claim
# :class: warning
#
# This short single-chain run is enough to connect code with a sampled
# posterior because the exact grid is available as a check. Real parameter
# estimation uses multiple chains or independent runs and checks divergences,
# effective sample size, convergence, and missed modes. The
# [sampler extension lab](01b_bayesian_samplers.ipynb) develops those ideas.
# :::

# %% [markdown]
# <details>
# <summary><i>Expected output &mdash; open this if your cell has not run yet</i></summary>
#
# <img src="https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/assets/expected/basics-numpyro-posterior.png" alt="expected output: basics-numpyro-posterior" style="max-width:100%">
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
# ## 5. The gravitational-wave bridge: the same likelihood in frequency
#
# Nothing fundamentally new happens here. In Sections 1--4 we assumed
#
# $$
# d_i\sim\mathcal N\!\left(h_i(\theta),\sigma^2\right).
# $$
#
# Every time sample had the same independent variance. That is **white noise**:
# its variance is spread uniformly over frequency, so its power spectral density
# (PSD) is flat.
#
# Real detector noise is coloured. Nearby time samples are then correlated, but
# for a stationary Gaussian process the Fourier coefficients are approximately
# independent. The same Gaussian likelihood becomes
#
# $$
# \widetilde d_k\sim
# \mathcal{CN}\!\left(\widetilde h_k(\theta),
# \frac{S_n(f_k)}{4\,\Delta f}\right),
# $$
#
# where $S_n(f)$ is the one-sided PSD. In words:
#
# > each Fourier mode is Gaussian around signal + noise, and the PSD tells us
# > the expected noise variance of that mode.
#
# This is the precise version of the “noisy pixels” picture. Raw time samples or
# image pixels have a covariance matrix; the PSD is the variance map only after
# transforming a stationary process into Fourier modes.

# %% [markdown]
# ### From residuals to the Whittle likelihood
#
# For residual $\widetilde r_k=\widetilde d_k-\widetilde h_k(\theta)$,
#
# $$
# \log \mathcal L(d\mid\theta,S_n)
# \simeq -\frac{1}{2}\sum_k
# \left[
# \frac{4\,\Delta f\,|\widetilde r_k|^2}{S_n(f_k)}
# +\log S_n(f_k)
# \right]+C.
# $$
#
# When the PSD is fixed, this is often written
#
# $$
# \log\mathcal L=-\frac12(d-h\mid d-h)+C,\qquad
# (a\mid b)=4\,\mathrm{Re}\sum_k
# \frac{\widetilde a_k\widetilde b_k^*}{S_n(f_k)}\Delta f.
# $$
#
# Compare this with the first Gaussian likelihood: residual squared divided by
# variance. The only change is that each frequency receives its own variance.
# Quiet frequencies receive more weight; noisy frequencies receive less.

# %% [markdown]
# **Predict before running:** The raw trace below contains a chirp and strong
# low-frequency noise. After division by the noise ASD, which feature should
# become easier to see: the low-frequency wander or the chirp?

# %% fqcp_figure="basics-psd-whitening-bridge"
from scipy.signal import chirp, spectrogram, welch, windows

bridge_rate = 512
bridge_duration = 8.0
bridge_time = np.arange(0, bridge_duration, 1 / bridge_rate)
bridge_frequency = np.fft.rfftfreq(bridge_time.size, 1 / bridge_rate)

# A smooth toy PSD shape: loud at low frequency, quiet in the middle, then rising.
noise_psd_shape = (
    1
    + (45 / np.maximum(bridge_frequency, 1)) ** 4
    + (bridge_frequency / 220) ** 2
)
noise_psd_shape[0] = noise_psd_shape[1]

white_noise = rng.normal(size=bridge_time.size)
coloured_noise = np.fft.irfft(
    np.fft.rfft(white_noise) * np.sqrt(noise_psd_shape),
    n=bridge_time.size,
)
coloured_noise /= np.std(coloured_noise)

bridge_signal = 0.32 * windows.tukey(bridge_time.size, alpha=0.35) * chirp(
    bridge_time,
    f0=8,
    f1=180,
    t1=bridge_duration,
    method="quadratic",
)
bridge_data = coloured_noise + bridge_signal
whitened_data = np.fft.irfft(
    np.fft.rfft(bridge_data) / np.sqrt(noise_psd_shape),
    n=bridge_time.size,
)
whitened_signal = np.fft.irfft(
    np.fft.rfft(bridge_signal) / np.sqrt(noise_psd_shape),
    n=bridge_time.size,
)

psd_frequency, estimated_psd = welch(
    coloured_noise,
    fs=bridge_rate,
    nperseg=1024,
    average="median",
)

fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
axes[0, 0].plot(bridge_time, bridge_data, lw=0.8, label="data")
axes[0, 0].plot(bridge_time, bridge_signal, lw=1.5, label="signal")
axes[0, 0].set(
    xlabel="time [s]",
    ylabel="amplitude [toy units]",
    title="Time series: signal + coloured noise",
)
axes[0, 0].legend()

axes[0, 1].loglog(psd_frequency[1:], estimated_psd[1:], color="C3")
axes[0, 1].set(
    xlabel="frequency [Hz]",
    ylabel="PSD [toy units$^2$/Hz]",
    title="Noise variance depends on frequency",
)

for ax, series, title in [
    (axes[1, 0], bridge_data, "Before PSD weighting"),
    (axes[1, 1], whitened_data, "After whitening by $\sqrt{S_n(f)}$"),
]:
    spec_f, spec_t, power = spectrogram(
        series,
        fs=bridge_rate,
        nperseg=256,
        noverlap=224,
    )
    band = (spec_f >= 5) & (spec_f <= 220)
    decibels = 10 * np.log10(power[band] + 1e-20)
    ax.pcolormesh(
        spec_t,
        spec_f[band],
        decibels,
        shading="auto",
        cmap="magma",
        vmin=np.percentile(decibels, 10),
        vmax=np.percentile(decibels, 99.5),
    )
    ax.set(xlabel="time [s]", ylabel="frequency [Hz]", title=title)

plt.show()

# %% [markdown]
# ### Code studio: whiten one data vector
#
# Complete the one Fourier-domain operation below. Divide each Fourier
# coefficient by the square root of its expected noise power, then run the
# check.

# %%
student_whitened_frequency = None  # YOUR CODE HERE

if student_whitened_frequency is None:
    print("Your turn: whiten np.fft.rfft(bridge_data) using noise_psd_shape.")
else:
    expected = np.fft.rfft(bridge_data) / np.sqrt(noise_psd_shape)
    np.testing.assert_allclose(student_whitened_frequency, expected)
    student_whitened_time = np.fft.irfft(
        student_whitened_frequency,
        n=bridge_time.size,
    )
    print("check passed:", student_whitened_time.shape)

# %% [markdown]
# <details>
# <summary>Show one possible solution</summary>
#
# ```python
# student_whitened_frequency = (
#     np.fft.rfft(bridge_data) / np.sqrt(noise_psd_shape)
# )
# ```
#
# </details>

# %% [markdown]
# Whitening does not manufacture a signal. It changes coordinates so that every
# Fourier mode is measured in units of its expected noise:
#
# $$
# \widetilde d_w(f)=\frac{\widetilde d(f)}{\sqrt{S_n(f)}}.
# $$
#
# The bright track is now easier to see because the low-frequency noise wall no
# longer dominates the scale. The LVK notebooks keep this logic but replace the
# toy chirp and toy PSD with physical waveforms, detector data, and an
# off-source PSD estimate.
#
# :::{admonition} Boundary
# :class: warning
#
# This diagonal Fourier-bin likelihood assumes approximate stationarity and
# well-behaved data. Gaps, glitches, strong lines, and time-varying noise create
# correlations that a single diagonal PSD does not describe.
# :::

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
# For a comparison of NUTS, Metropolis, nested sampling, and variational
# inference, continue to the independently runnable
# [sampler extension lab](01b_bayesian_samplers.ipynb).
# :::

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
