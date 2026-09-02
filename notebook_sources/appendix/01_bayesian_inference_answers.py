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
# ---
#
# # Worked solutions
#
# Everything above this line is the lab notebook, unchanged. Everything below is
# the instructor version of Section 5. Run the notebook from the top so that
# `fit_on_grid`, `run_nuts`, `interval`, `posterior_predictive`, `data`, `time`,
# and `sigma` are all defined.
#
# Exercises 1 and 2 are solved with the **sampler**, because that is the tool
# that survives past two parameters. Exercises 3 and 4 need evidences, so they
# fall back to the **grid** — and the exercise-4 discussion says why that is not
# a coincidence.

# %% [markdown]
# ## Exercise 1 — does more data help?
#
# Thin the data evenly so that only $N$ changes. Both tools take `time` as a
# keyword argument, so each refit is one line. We use the sampler here.

# %%
LINE_BOUNDS = ((-2.0, 6.0), (-15.0, 15.0))
counts = (10, 25, 50, 100)


def slope_width(draws):
    """90% credible width of the slope."""
    low, high = np.quantile(draws["a"], [0.05, 0.95])
    return high - low


fig, ax = plt.subplots(figsize=(7.5, 3.4))
widths = {}
for n_used in counts:
    keep = np.linspace(0, data.size - 1, n_used).astype(int)
    draws = run_nuts(
        signal_model, LINE_BOUNDS, observed=data[keep], time=time[keep]
    )
    widths[n_used] = slope_width(draws)
    ax.hist(
        draws["a"],
        bins=40,
        density=True,
        histtype="step",
        lw=1.6,
        label=f"N = {n_used:3d}   90% width {widths[n_used]:.2f}",
    )
ax.set(xlabel="slope m", ylabel="marginal posterior", xlim=(1.0, 3.5))
ax.legend(fontsize=8)
ax.set_title("Evenly thinned data: only N changes")
plt.show()

reference = widths[10] / np.sqrt(np.array(counts) / 10)
for n_used, predicted in zip(counts, reference):
    print(
        f"N = {n_used:3d}   width {widths[n_used]:.3f}   "
        f"1/sqrt(N) prediction {predicted:.3f}"
    )

# The first-N-points version, for the follow-up question.
for n_used in counts:
    draws = run_nuts(
        signal_model, LINE_BOUNDS, observed=data[:n_used], time=time[:n_used]
    )
    print(f"first {n_used:3d} points only: width {slope_width(draws):.3f}")

# %% [markdown]
# **What to say out loud.** With the baseline held fixed the width tracks
# $1/\sqrt N$, the expected scaling for independent Gaussian observations. It is
# *not* guaranteed to be monotonic: a single unlucky point can widen the
# posterior, and it does not have to shrink for one particular noise
# realisation. The theorem is about the average over realisations, not about
# this dataset.
#
# The first-$N$-points version contracts much faster than $1/\sqrt N$, and that
# is the more interesting half of the exercise. Two things changed at once:
# the number of points, and the length of the lever arm. A slope measured over
# $t\in[0,1]$ is poorly determined no matter how many points sit in that
# interval; stretching to $t\in[0,10]$ improves it by a further factor of the
# baseline. Roughly, $\sigma_m \propto 1/(T\sqrt N)$.
#
# The lesson generalises directly: **how much information a dataset carries
# depends on where the samples are, not only on how many there are.** This is
# why gravitational-wave sensitivity depends on bandwidth and observing time and
# not just on sample rate, and why LISA cares so much about mission duration.
#
# Note also what does *not* improve with $N$: the model is still the wrong shape.
# More data makes a wrong answer more precise, not more correct — though
# Exercise 3 shows it does make the wrongness easier to detect.

# %% [markdown]
# ## Exercise 2 — what if the assumed noise is wrong?

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for assumed, label in [
    (sigma / 2, r"assumed $\sigma/2$ (too small)"),
    (sigma, r"assumed $\sigma$ (correct)"),
    (2 * sigma, r"assumed $2\sigma$ (too large)"),
]:
    draws = run_nuts(signal_model, LINE_BOUNDS, sigma=assumed)
    ax.hist(
        draws["a"],
        bins=40,
        density=True,
        histtype="step",
        lw=1.6,
        label=f"{label}: 90% width {slope_width(draws):.2f}",
    )
ax.set(xlabel="slope m", ylabel="marginal posterior", xlim=(1.5, 3.0))
ax.legend(fontsize=8)
ax.set_title("Same data, three noise assumptions")
plt.show()

# %% [markdown]
# **What to say out loud.** Halving the assumed $\sigma$ halves the posterior
# width. Nothing about the data changed; the likelihood was simply told that each
# point is four times more informative than it is, so it believed the scatter
# more than it should have.
#
# The posterior width is therefore **not** a measurement of how much the data
# know. It is a measurement of how much the data know *given the noise model you
# asserted*. An underestimated PSD produces a confident, wrong, and entirely
# self-consistent answer — the posterior itself never complains.
#
# The check that does complain is the posterior predictive one: with
# $\sigma/2$ assumed, the replicated datasets are far too tight to look like the
# real one.
#
# This is exactly why Parts 2 and 3 of the course spend so much effort on PSD
# estimation, off-source noise windows, and PSD uncertainty. In a
# gravitational-wave likelihood, $S_n(f)$ *is* the assumed noise.

# %% [markdown]
# ## Exercise 3 — try a different signal model
#
# This one needs an evidence, so it goes back to the grid.

# %%
def exponential_model(time, amplitude, tau):
    return amplitude * (jnp.exp(time / tau) - 1.0)


a_grid = np.linspace(0, 4, 141)
tau_grid = np.linspace(1, 10, 161)
A, T, exp_posterior, log_z_exponential = fit_on_grid(
    exponential_model, data, a_grid, tau_grid
)

exp_curves, exp_replicas, exp_p = posterior_predictive(
    exponential_model, A, T, exp_posterior, data
)
line_curves, line_replicas, line_p = posterior_predictive(
    signal_model, M, C, posterior, data
)

print(f"log Z (line)       : {log_z_line:9.2f}   posterior predictive p = {line_p:.3f}")
print(
    f"log Z (exponential): {log_z_exponential:9.2f}   "
    f"posterior predictive p = {exp_p:.3f}"
)
print(f"log Bayes factor, exponential over line: {log_z_exponential - log_z_line:.1f}")

line_volume = (m_grid[-1] - m_grid[0]) * (c_grid[-1] - c_grid[0])
exp_volume = (a_grid[-1] - a_grid[0]) * (tau_grid[-1] - tau_grid[0])
print(f"prior-volume difference alone is worth {np.log(line_volume / exp_volume):.1f}")

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
for column, (name, curves, replicas, p_value) in zip(
    axes.T,
    [
        ("line (wrong shape)", line_curves, line_replicas, line_p),
        ("exponential (true shape)", exp_curves, exp_replicas, exp_p),
    ],
):
    top, bottom = column
    top.fill_between(
        time, *np.quantile(replicas, [0.05, 0.95], axis=0), alpha=0.2, color="C0"
    )
    top.fill_between(
        time, *np.quantile(curves, [0.05, 0.95], axis=0), alpha=0.5, color="C1"
    )
    top.plot(time, data, "o", ms=3, color="k")
    top.set_title(f"{name}\np = {p_value:.3f}")

    bottom.axhline(0, color="k", lw=1)
    bottom.fill_between(
        time,
        *np.quantile(replicas - curves, [0.05, 0.95], axis=0),
        alpha=0.25,
        color="C0",
    )
    bottom.plot(time, np.median(data - curves, axis=0), color="C3")
    bottom.set(xlabel="time", ylim=(-12, 12))
axes[0, 0].set_ylabel("observation")
axes[1, 0].set_ylabel("data - signal")
plt.show()

# %% [markdown]
# **What to say out loud.** The exponential wins by tens of log units, and the
# prior-volume difference between the two rectangles accounts for under 2 of
# that. So the preference is driven by the fit, not by the bookkeeping —
# but note that a log Bayes factor of 30 and a log Bayes factor of 3 would need
# very different amounts of care about that bookkeeping.
#
# The residual panels are the more convincing evidence. The line's residual is a
# coherent arc; the exponential's is scatter around zero, and its $p$-value is
# unremarkable.
#
# The two diagnostics agree here, and they are genuinely independent: the
# evidence is a comparison *between* models, the $p$-value is a falsification
# test of *one* model against its own predictions. The evidence would still name
# a winner if every model on the menu were wrong. The $p$-value is what notices
# that case.

# %% [markdown]
# ## Exercise 4 — a model that is wrong in a different way

# %%
def sine_model(time, amplitude, frequency):
    return amplitude * jnp.sin(2 * np.pi * frequency * time)


amplitude_grid = np.linspace(0, 40, 141)
frequency_grid = np.linspace(0.01, 1.0, 161)
S_a, S_f, sine_posterior, log_z_sine = fit_on_grid(
    sine_model, data, amplitude_grid, frequency_grid
)
_, _, sine_p = posterior_predictive(sine_model, S_a, S_f, sine_posterior, data)

rows = [
    ("noise only", log_z_noise, None),
    ("sinusoid", log_z_sine, sine_p),
    ("line", log_z_line, line_p),
    ("exponential", log_z_exponential, exp_p),
]
print(f"{'model':14s} {'log Z':>10s} {'log B vs noise':>16s} {'ppc p':>8s}")
for name, log_z, p_value in rows:
    p_text = "     -  " if p_value is None else f"{p_value:8.3f}"
    print(f"{name:14s} {log_z:10.1f} {log_z - log_z_noise:16.1f} {p_text}")

# %% [markdown]
# **What to say out loud.** Every signal model on the list beats noise-only by
# several hundred in log evidence, including the sinusoid, which is not remotely
# the right shape. "Beats noise-only decisively" is therefore a very low bar. It
# is worth stating plainly: **the detection statistic and the model-selection
# statistic are not the same question**, even though both are log Bayes factors.
#
# The two rankings agree — exponential, then line, then sinusoid — which is
# reassuring but not guaranteed. Evidence rewards a model for predicting the data
# well *on average over its prior*; the posterior predictive $p$-value asks
# whether the best-fitting version of the model could have produced data like
# these at all. A model can win on evidence and still fail its own predictive
# check, and when it does, the honest report is that none of the candidates is
# adequate.
#
# The honest statement about the winner is the one that survives to the
# gravitational-wave chapters:
#
# > Of the models we tried, the exponential is strongly preferred, and it is not
# > excluded by its own posterior predictive check.
#
# Not "the signal is an exponential". Here the true shape happened to be on the
# menu. In a real analysis it never is: waveform models are approximations, and
# that gap is what waveform-systematics studies exist to bound.

# %% [markdown]
# ## The one-slide summary
#
# | question | tool | what it cannot do |
# | --- | --- | --- |
# | what are the parameters? | posterior | tell you the model is wrong |
# | how confident should I be? | posterior width | protect you from a wrong noise model |
# | is there anything here at all? | $\log\mathcal B$ vs noise | tell you *what* is here |
# | which of my models is best? | $\log\mathcal B$ between models | tell you any of them is adequate |
# | is my best model adequate? | posterior predictive check | prove that it is right |
#
# The rest of the course changes the data, the response, and the noise model.
# This table does not change.
