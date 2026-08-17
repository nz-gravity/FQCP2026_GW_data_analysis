"""Build the three Colab-first FQCP 2026 gravitational-wave PE notebooks."""

from pathlib import Path
from textwrap import dedent

import black
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks"


def clean_source(text):
    """Remove generator indentation and surrounding blank lines."""
    return dedent(text).strip()


def md(text):
    """Create Markdown with display-math delimiters supported everywhere."""
    source = clean_source(text)
    source = source.replace(r"\[", "$$").replace(r"\]", "$$")
    return nbf.v4.new_markdown_cell(source)


def code(text):
    """Create a consistently formatted, reader-facing Python cell."""
    source = clean_source(text)
    source = black.format_str(source, mode=black.Mode(line_length=88)).rstrip()
    return nbf.v4.new_code_cell(source)


def write(name, title, cells):
    header = md(f"""# {title}

**FQCP 2026 · Bayesian parameter estimation for gravitational-wave sources**

> Google Colab worksheet for early-stage graduate students. Run from top to
> bottom; **Extension** sections may be skipped live.
""")
    notebook = nbf.v4.new_notebook(cells=[header, *cells])
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": name, "provenance": []},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUT / name)
    print("Wrote", OUT / name)


OUT.mkdir(parents=True, exist_ok=True)
for old_notebook in OUT.glob("*.ipynb"):
    old_notebook.unlink()

STANDARD_SETUP = code("""import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
rng = np.random.default_rng(20260817)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["animation.html"] = "jshtml"
print("Running in Colab:", IN_COLAB)""")


write(
    "00_basics_parameter_estimation.ipynb",
    "Basics: what is parameter estimation?",
    [
        md(r"""## Goal

Parameter estimation (PE) means learning about unknown parameters from data. By
the end of this notebook you should be able to distinguish the four pieces of
Bayes' theorem, calculate a small posterior by hand, and explain why a noise PSD
appears in a gravitational-wave likelihood.

We will follow the teaching sequence used in the NZ Bilby CBC workshop:

1. write a signal model;
2. choose priors;
3. write a likelihood from a noise assumption;
4. calculate a posterior on a grid;
5. inspect marginals and posterior predictions;
6. replace the grid with the algorithms real analyses use — Metropolis-Hastings,
   the Fisher approximation, and nested sampling — and learn how to tell whether
   they worked;
7. replace white noise by a gravitational-wave-style PSD-weighted likelihood.

Sections 5-7 and the extensions are written to be read on your own afterwards;
they are the reference half of this notebook and are not all covered live.

Bayes' theorem is

\[
p(\theta\mid d,M)=\frac{p(d\mid\theta,M)\,p(\theta\mid M)}{p(d\mid M)}
=\frac{\mathcal L(d\mid\theta)\,\pi(\theta)}{\mathcal Z}.
\]

| Quantity | Meaning | Depends on |
| --- | --- | --- |
| prior $\pi(\theta)$ | what parameter values the model allows before these data | assumptions and previous information |
| likelihood $\mathcal L(d\mid\theta)$ | how compatible the observed data are with a proposed parameter value | signal and noise models |
| posterior $p(\theta\mid d)$ | updated uncertainty after conditioning on the observed data | prior × likelihood |
| evidence $\mathcal Z=p(d\mid M)$ | average likelihood across the prior for one complete model | likelihood and prior volume |

The evidence normalises the posterior and can compare complete models through a
Bayes factor. It is not a parameter estimate or a generic goodness-of-fit score.

### A posterior is more than one best-fit template

A maximum-likelihood template answers “which tested parameter value fits best?”
A posterior also exposes uncertainty, degeneracies, multiple modes, and prior
sensitivity, and it can be propagated into predictions or population analyses.

This is not the same as saying that frequentist inference only produces point
estimates: confidence regions, profile likelihoods, and sampling distributions
also quantify uncertainty. The key distinction is interpretation. A Bayesian
credible interval assigns probability to parameters conditional on the model
and observed data; a frequentist confidence procedure is calibrated over
repeated hypothetical datasets."""),
        STANDARD_SETUP,
        md(
            r"""## 1. Data and a signal model

Assume $d_i=m t_i+c+n_i$ and independent Gaussian noise $n_i\sim\mathcal N(0,\sigma^2)$. Every likelihood statement is conditional on assumptions like these."""
        ),
        code(
            """true_parameters={"m":.5,"c":.2}; sigma=3.0
time=np.linspace(0,10,100)
def signal_model(time,m,c): return m*time+c
data=signal_model(time,**true_parameters)+rng.normal(0,sigma,time.size)
fig,ax=plt.subplots(figsize=(8,3.3)); ax.plot(time,data,"o",ms=3,label="data")
ax.plot(time,signal_model(time,**true_parameters),lw=2,label="injected signal")
ax.set(xlabel="time",ylabel="observation",title="Data = signal + noise"); ax.legend(); plt.show()"""
        ),
        md(
            r"""## 2. Priors and prior predictive checks

Take $m\sim\mathrm{Uniform}(0,1.5)$ and $c\sim\mathrm{Uniform}(-5,5)$. A prior is part of the model, not an afterthought. Drawing curves from it checks whether the model can plausibly describe the data before inference."""
        ),
        code(
            """n_prior=2500
prior_m=rng.uniform(0,1.5,n_prior); prior_c=rng.uniform(-5,5,n_prior)
fig,axes=plt.subplots(1,2,figsize=(10,3.4))
axes[0].hist(prior_m,bins=30,density=True,histtype="step",label="m")
axes[0].hist(prior_c,bins=30,density=True,histtype="step",label="c")
axes[0].set(xlabel="parameter value",ylabel="prior density",title="Marginal priors"); axes[0].legend()
axes[1].plot(time,data,"o",ms=3,color="k")
for m,c in zip(prior_m[:40],prior_c[:40]): axes[1].plot(time,signal_model(time,m,c),color="C0",alpha=.08)
axes[1].set(xlabel="time",ylabel="observation",title="Prior predictive curves"); plt.show()"""
        ),
        md(
            r"""## 3. Gaussian likelihood

\[
\log\mathcal L(d\mid m,c)=-\frac12\sum_i\left[
\frac{(d_i-mt_i-c)^2}{\sigma^2}+\log(2\pi\sigma^2)\right].
\]

Changing the assumed noise scale changes the width of the posterior. If the noise model is wrong, a mathematically correct sampler still gives a misleading answer."""
        ),
        code("""def log_likelihood(m,c):
    residual=data-signal_model(time,m,c)
    return -.5*np.sum((residual/sigma)**2+np.log(2*np.pi*sigma**2))

m_grid=np.linspace(0,1.5,141); c_grid=np.linspace(-5,5,161)
M,C=np.meshgrid(m_grid,c_grid,indexing="ij")
logL=np.array([[log_likelihood(m,c) for c in c_grid] for m in m_grid])
log_prior=np.zeros_like(logL)  # constant inside this finite grid
log_posterior=logL+log_prior
posterior=np.exp(log_posterior-log_posterior.max())
posterior/=np.trapezoid(np.trapezoid(posterior,c_grid,axis=1),m_grid)

fig,axes=plt.subplots(1,3,figsize=(12,3.4),sharex=True,sharey=True)
for ax,values,title in zip(axes,[np.exp(log_prior),np.exp(logL-logL.max()),posterior],["prior","likelihood","posterior"]):
    image=ax.contourf(m_grid,c_grid,values.T,levels=24,cmap="magma")
    ax.plot(true_parameters["m"],true_parameters["c"],"c*",ms=10); ax.set(title=title,xlabel="slope m")
axes[0].set_ylabel("intercept c"); plt.show()"""),
        md(
            """The posterior is a ridge: increasing the slope can be compensated by decreasing the intercept. Marginalisation integrates over the other parameter; it is not the same as holding it at a best-fit value."""
        ),
        code(
            """p_m=np.trapezoid(posterior,c_grid,axis=1); p_c=np.trapezoid(posterior,m_grid,axis=0)
def interval(grid,density):
    cdf=np.r_[0,np.cumsum((density[:-1]+density[1:])*np.diff(grid)/2)]; cdf/=cdf[-1]
    return np.interp([.05,.5,.95],cdf,grid)
fig,axes=plt.subplots(1,2,figsize=(9,3.2))
for ax,grid,density,name,truth in zip(axes,[m_grid,c_grid],[p_m,p_c],["m","c"],true_parameters.values()):
    q=interval(grid,density); ax.plot(grid,density); ax.axvline(truth,color="k",ls="--")
    ax.axvspan(q[0],q[2],alpha=.2); ax.set(xlabel=name,ylabel="marginal posterior",title=f"median {q[1]:.2f}; 90% [{q[0]:.2f}, {q[2]:.2f}]")
plt.show()"""
        ),
        md(r"""### What does the evidence do?

For a model $M$, the evidence averages the likelihood over its **normalised**
prior,

\[
\mathcal Z_M=\int \mathcal L(d\mid\theta,M)\,\pi(\theta\mid M)\,d\theta.
\]

The next cell compares a line with free slope and intercept ($M_1$) against a
line forced through zero ($M_0$). A model does not win merely because its best
fit is higher: extra prior volume that fits poorly reduces its evidence. This
is the Bayesian form of an Occam penalty, and it also means Bayes factors must
be reported with their priors."""),
        code('''def log_trapezoid_exp(log_values, grid, axis=-1):
    """Stable log of the trapezoidal integral of exp(log_values)."""
    reference = np.max(log_values)
    integral = np.trapezoid(np.exp(log_values - reference), grid, axis=axis)
    return reference + np.log(integral)


# M1: m and c are both free with a uniform prior on the plotted rectangle.
log_z_free_intercept = (
    log_trapezoid_exp(
        np.array(
            [log_trapezoid_exp(logL[row], c_grid) for row in range(len(m_grid))]
        ),
        m_grid,
    )
    - np.log((m_grid[-1] - m_grid[0]) * (c_grid[-1] - c_grid[0]))
)

# M0: c=0 exactly and only m is free.
logL_zero_intercept = np.array([log_likelihood(m, 0.0) for m in m_grid])
log_z_zero_intercept = log_trapezoid_exp(logL_zero_intercept, m_grid) - np.log(
    m_grid[-1] - m_grid[0]
)

log_bayes_factor = log_z_free_intercept - log_z_zero_intercept
print(f"log Z (free intercept): {log_z_free_intercept:.2f}")
print(f"log Z (zero intercept): {log_z_zero_intercept:.2f}")
print(f"log Bayes factor, free/zero: {log_bayes_factor:.2f}")'''),
        md(
            """### Fast animation: information accumulates

Each frame uses a longer prefix of the same dataset. The posterior does not have to shrink monotonically for every noise realisation, but its typical scale contracts as information accumulates."""
        ),
        code("""model_cube=M[:,:,None]*time[None,None,:]+C[:,:,None]
cumulative_sse=np.cumsum((data[None,None,:]-model_cube)**2,axis=2)
frame_sizes=np.arange(8,time.size+1,6); slope_densities=[]
for n_used in frame_sizes:
    frame_logp=-.5*cumulative_sse[:,:,n_used-1]/sigma**2
    frame_p=np.exp(frame_logp-frame_logp.max()); marginal=np.trapezoid(frame_p,c_grid,axis=1)
    slope_densities.append(marginal/np.trapezoid(marginal,m_grid))
fig,ax=plt.subplots(figsize=(7,3.2)); line,=ax.plot([],[],color="C3"); ax.axvline(true_parameters["m"],color="k",ls="--")
ax.set(xlim=(m_grid.min(),m_grid.max()),ylim=(0,1.1*np.max(slope_densities)),xlabel="slope m",ylabel="posterior density")
def animate_learning(i):
    line.set_data(m_grid,slope_densities[i]); ax.set_title(f"posterior after {frame_sizes[i]} observations"); return (line,)
learning_animation=FuncAnimation(fig,animate_learning,frames=len(frame_sizes),interval=150)
plt.close(fig); display(HTML(learning_animation.to_jshtml()))"""),
        md(
            """## 4. Posterior predictive check

Draw parameter pairs from the posterior and map each through the signal model. This asks whether the inferred model can reproduce data like those observed."""
        ),
        code(
            """weights=(posterior/posterior.sum()).ravel(); choices=rng.choice(weights.size,size=250,replace=True,p=weights)
m_samples=M.ravel()[choices]; c_samples=C.ravel()[choices]
predictions=np.array([signal_model(time,m,c) for m,c in zip(m_samples,c_samples)])
low,median,high=np.quantile(predictions,[.05,.5,.95],axis=0)
fig,ax=plt.subplots(figsize=(8,3.3)); ax.plot(time,data,"o",ms=3,color="k",label="data")
ax.plot(time,median,label="posterior median"); ax.fill_between(time,low,high,alpha=.25,label="90% signal band")
ax.set(xlabel="time",ylabel="observation",title="Posterior predictive signal"); ax.legend(); plt.show()"""
        ),
        md(r"""## 5. Why real PE cannot use a grid

Everything so far used a grid. That worked because the model had two
parameters. Grids die quickly: with $n$ points per axis and $D$ parameters, a
grid costs $n^D$ likelihood evaluations.

A binary black hole has about 15 parameters. At a coarse 20 points per axis
that is $20^{15}\approx3\times10^{19}$ waveform evaluations. At one
millisecond each, that is roughly a billion years.

Stochastic samplers escape this because they spend their effort where the
posterior actually has mass, rather than visiting the (overwhelmingly empty)
rest of the prior volume."""),
        code("""dimensions = np.arange(1, 16)
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
print(f"Fifteen parameters: {20**15:.2e} evaluations")"""),
        md(r"""### Metropolis-Hastings in twelve lines

The Metropolis algorithm needs only the ability to *evaluate* the unnormalised
posterior $\mathcal L(\theta)\pi(\theta)$; it never needs the evidence. From
the current point $\theta$:

1. propose $\theta'=\theta+\mathcal N(0,\Sigma_{\rm prop})$;
2. accept with probability
   $\min\left[1,\dfrac{\mathcal L(\theta')\pi(\theta')}{\mathcal L(\theta)\pi(\theta)}\right]$;
3. if rejected, **record the current point again**.

Step 3 is not a bug. Rejections are how the chain builds up density in regions
of high posterior probability. The resulting chain is a set of correlated draws
whose histogram converges to the posterior."""),
        code('''PRIOR_BOX = np.array([[0.0, 1.5], [-5.0, 5.0]])  # rows: m, c


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
print(f"chain shape: {chain.shape}")'''),
        md(
            """### Animation: watch the chain find the posterior

The walker starts in a corner where the posterior is negligible. The first
phase is **burn-in**: a directed climb towards the bulk of the probability.
Only afterwards does the chain wander around the degeneracy ridge in the way
that actually samples it. Burn-in samples are discarded because they depend on
where you started, not on the posterior."""
        ),
        code("""frame_steps = np.arange(20, 1400, 26)
fig, (walk_ax, trace_ax) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=72)
walk_ax.contour(m_grid, c_grid, posterior.T, levels=6, cmap="magma")
(path,) = walk_ax.plot([], [], lw=0.7, color="C0", alpha=0.8)
(head,) = walk_ax.plot([], [], "o", color="C3", ms=7)
walk_ax.plot(true_parameters["m"], true_parameters["c"], "c*", ms=12)
walk_ax.set(
    xlim=(0, 1.5), ylim=(-5, 5), xlabel="slope m", ylabel="intercept c"
)
(trace_line,) = trace_ax.plot([], [], lw=0.8, color="C0")
trace_ax.axhline(true_parameters["m"], color="k", ls="--")
trace_ax.set(
    xlim=(0, frame_steps[-1]), ylim=(0, 1.5), xlabel="step", ylabel="slope m"
)


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
display(HTML(chain_animation.to_jshtml()))"""),
        md(
            """### The proposal scale controls everything

A chain can be perfectly correct in principle and useless in practice. Too
small a step and the walker crawls, accepting almost everything but exploring
nothing. Too large and almost every proposal lands somewhere absurd and is
rejected. Both failures produce a chain that has not forgotten its starting
point, and the "too small" chain below still reports a badly wrong mean.

A useful rule of thumb for random-walk Metropolis is an acceptance fraction
near 0.2-0.3. Production samplers (`emcee`, `dynesty`, `bilby`'s defaults)
automate this tuning, but the failure modes remain the same."""
        ),
        code("""settings = [
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
        title=f"{label}\\nacceptance {trial_acceptance:.2f}",
    )
    print(
        f"{label:>10}: acceptance {trial_acceptance:.2f}, "
        f"posterior mean m = {trial_chain[500:, 0].mean():.3f}"
    )
axes[0].set_ylabel("slope m")
axes[0].set_ylim(0, 1.5)
plt.show()"""),
        md(r"""### Diagnostics: burn-in and effective sample size

Consecutive Metropolis samples are correlated, so $N$ stored samples are worth
fewer than $N$ independent draws. The autocorrelation function
$\rho(k)$ measures this, and the effective sample size is

\[
N_{\rm eff}\simeq\frac{N}{1+2\sum_{k\ge1}\rho(k)}.
\]

$N_{\rm eff}$, not the raw chain length, sets the Monte Carlo error on any
posterior summary. A chain of a million highly correlated samples can carry
less information than a thousand independent ones."""),
        code('''burn_in = 500
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
axes[1].set(xlabel="lag [steps]", ylabel=r"$\\rho$", title="Autocorrelation")
axes[1].legend()
plt.show()

for index, name in enumerate(["m", "c"]):
    print(
        f"{name}: N = {samples.shape[0]}, "
        f"N_eff = {effective_sample_size(samples[:, index]):.0f}"
    )'''),
        md(
            """### The corner plot, and a check against the grid

A corner plot is the standard way to display a multi-dimensional posterior: 1D
marginals on the diagonal, 2D marginals below. Because this problem is small
enough to solve both ways, we can overlay the exact grid marginals in orange.
Agreement is the check that the sampler is doing its job, and it is the only
reason to trust the sampler on problems where no grid is possible."""
        ),
        code("""import subprocess
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
print(f"sampler : m = {samples[:, 0].mean():.4f}")"""),
        md(r"""## 6. The Fisher matrix: a cheap Gaussian approximation

Expanding $\log\mathcal L$ to second order about its maximum approximates the
posterior by a Gaussian with covariance $F^{-1}$, where

\[
F_{ij}=-\left\langle\frac{\partial^2\log\mathcal L}
{\partial\theta_i\partial\theta_j}\right\rangle .
\]

For our linear model with Gaussian noise this is not an approximation at all:
with a flat prior the posterior *is* exactly Gaussian, and
$F=X^{\mathsf T}X/\sigma^2$ for the design matrix $X$. That makes it a clean
place to see what the Fisher matrix does before trusting it elsewhere.

In gravitational-wave work the same object appears as $F_{ij}=(\partial_i h\mid\partial_j h)$
and is widely used for forecasts. Be careful: it is only reliable at high
signal-to-noise ratio and for near-linear models. It cannot see multiple modes,
hard prior boundaries, or curved (banana-shaped) degeneracies."""),
        code("""design_matrix = np.column_stack([time, np.ones_like(time)])
fisher_matrix = design_matrix.T @ design_matrix / sigma**2
fisher_covariance = np.linalg.inv(fisher_matrix)
fisher_mean = fisher_covariance @ design_matrix.T @ data / sigma**2

fisher_sd = np.sqrt(np.diag(fisher_covariance))
correlation = fisher_covariance[0, 1] / (fisher_sd[0] * fisher_sd[1])

angles = np.linspace(0, 2 * np.pi, 200)
eigenvalues, eigenvectors = np.linalg.eigh(fisher_covariance)
circle = np.column_stack([np.cos(angles), np.sin(angles)])

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.contour(m_grid, c_grid, posterior.T, levels=6, cmap="magma")
for n_sigma in (1, 2):
    ellipse = fisher_mean + n_sigma * circle @ (
        eigenvectors * np.sqrt(eigenvalues)
    ).T
    ax.plot(ellipse[:, 0], ellipse[:, 1], color="C0", lw=2)
ax.plot(true_parameters["m"], true_parameters["c"], "c*", ms=12)
ax.set(
    xlim=fisher_mean[0] + 4 * np.array([-1, 1]) * fisher_sd[0],
    ylim=fisher_mean[1] + 4 * np.array([-1, 1]) * fisher_sd[1],
    xlabel="slope m",
    ylabel="intercept c",
    title="Fisher 1- and 2-sigma ellipses over the grid posterior",
)
plt.show()

print(f"Fisher   sd: m = {fisher_sd[0]:.4f}, c = {fisher_sd[1]:.4f}")
print(f"Sampler  sd: m = {samples[:, 0].std():.4f}, c = {samples[:, 1].std():.4f}")
print(f"m-c correlation coefficient: {correlation:+.3f}")"""),
        md(r"""## 7. Nested sampling: where the evidence comes from

Section 3 computed the evidence with a grid. Real analyses cannot. Nested
sampling reorganises the integral by *prior volume*: let $X(\lambda)$ be the
fraction of the prior with $\mathcal L>\lambda$. Then the $D$-dimensional
integral collapses to a one-dimensional one,

\[
\mathcal Z=\int \mathcal L\,\pi\,d\theta=\int_0^1\mathcal L(X)\,dX .
\]

The algorithm keeps $N_{\rm live}$ points drawn from the prior, repeatedly
deletes the worst one, and replaces it with a new point drawn from the prior
*subject to* $\mathcal L>\mathcal L_{\rm worst}$. Each deletion shrinks the
volume by a known factor, on average $X_i\approx e^{-i/N_{\rm live}}$, so the
deleted likelihoods and their volume shells accumulate into $\mathcal Z$. The
discarded points, suitably weighted, are posterior samples as a by-product.

This is what `dynesty`, `MultiNest`, and `PolyChord` do, and it is why Bayesian
model comparison is practical in gravitational-wave astronomy at all."""),
        code('''def nested_sampling(
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
print(f"difference           : {log_z_nested - log_z_free_intercept:+.3f}")'''),
        md(
            """### Animation: the live points contract onto the posterior

Each frame shows the surviving live points. They begin spread over the whole
prior and are squeezed into the high-likelihood ridge as the likelihood
threshold rises. The right panel shows the integrand $\\mathcal{L}(X)$ against
$\\log X$: the evidence is the area under it, and the visible bump is the
region of prior volume that actually contributes."""
        ),
        code("""fig, (live_ax, mass_ax) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=72)
live_ax.contour(m_grid, c_grid, posterior.T, levels=6, cmap="magma")
(live_points,) = live_ax.plot([], [], ".", color="C0", ms=3)
live_ax.set(xlim=(0, 1.5), ylim=(-5, 5), xlabel="slope m", ylabel="intercept c")

log_volume_axis = -np.arange(dead_logl.size) / 250
posterior_mass = np.exp(dead_logl + dead_logw - np.max(dead_logl + dead_logw))
mass_ax.plot(log_volume_axis, posterior_mass, color="0.7")
(mass_line,) = mass_ax.plot([], [], color="C3", lw=2)
mass_ax.set(
    xlabel=r"$\\log X$ (log prior volume)",
    ylabel=r"$\\mathcal{L}\\,\\Delta X$ (normalised)",
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
display(HTML(nested_animation.to_jshtml()))"""),
        md(r"""## 8. The gravitational-wave bridge: PSD and Whittle likelihood

A power spectral density (PSD) describes how a stationary random process's
variance is distributed over frequency. For one-sided $S_n(f)$,
$S_n(f)\,df$ is the expected noise variance in a small positive-frequency band.
Its units are strain$^2$/Hz; the amplitude spectral density (ASD)
$\sqrt{S_n(f)}$ has units strain/$\sqrt{\mathrm{Hz}}$.

For an approximately stationary, Gaussian time series, well-behaved Fourier
coefficients are approximately independent complex Gaussians. This gives the
Whittle approximation

\[
\log \mathcal L(d\mid\theta,S_n)
\simeq -\frac{1}{2}\sum_k
\left[\frac{4\,\Delta f\,|\tilde d_k-\tilde h_k(\theta)|^2}{S_n(f_k)}
+\log S_n(f_k)\right]+C.
\]

When the PSD is fixed, the $\log S_n$ term is constant and we often write

\[
\log\mathcal L=-\frac12(d-h\mid d-h)+C,\qquad
(a\mid b)=4\,\mathrm{Re}\sum_k
\frac{\tilde a_k\tilde b_k^*}{S_n(f_k)}\Delta f.
\]

The inverse PSD is therefore a frequency-dependent weight: residual power in a
quiet band matters more. Gaps, strong lines, spectral leakage, and
non-stationarity couple Fourier bins and weaken the simple independence
approximation."""),
        code("""from scipy.signal import welch

sample_rate = 512
duration = 32
noise_time = np.arange(0, duration, 1 / sample_rate)
noise_frequency = np.fft.rfftfreq(noise_time.size, 1 / sample_rate)

# A deliberately non-white spectrum: large low-frequency noise and a mild
# high-frequency rise. The absolute normalisation is arbitrary in this toy.
noise_shape = (
    1
    + (30 / np.maximum(noise_frequency, 1)) ** 4
    + (noise_frequency / 180) ** 2
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
    ylabel=r"ASD [toy units/$\\sqrt{\\mathrm{Hz}}$]",
    title="Welch estimate of the ASD",
)
plt.show()"""),
        md("""### Optional audio analogy: hear what whitening does

This is **not detector strain converted to sound**. It is an audible toy with a
chirp buried in coloured noise. The second clip divides each Fourier component
by the known noise ASD (“whitening”), the same inverse-noise idea that appears
in the Whittle likelihood."""),
        code("""from IPython.display import Audio
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
display(safe_audio(whitened_audio))"""),
        md(
            """## Extension: is the posterior actually calibrated?

A posterior can be self-consistent and still be wrong. The standard check is a
**probability-probability (P-P) test**: draw a truth from the prior, simulate
data, run inference, and record the quantile at which the truth falls in its
own posterior. If the analysis is correct those quantiles are uniform, so the
cumulative curve is a diagonal.

This is how the LVK collaboration validates parameter-estimation pipelines,
and it catches errors that no single analysis can reveal. Here the linear model
has an exact Gaussian posterior (Section 6), so hundreds of trials are cheap.
A deviating curve means a bug, a wrong noise model, or a prior mismatch."""
        ),
        code("""from scipy.stats import norm

n_trials = 400
calibration_rng = np.random.default_rng(11)
quantiles = []
for _ in range(n_trials):
    truth = np.array(
        [calibration_rng.uniform(0, 1.5), calibration_rng.uniform(-5, 5)]
    )
    trial_data = signal_model(time, *truth) + calibration_rng.normal(
        0, sigma, time.size
    )
    estimate = fisher_covariance @ design_matrix.T @ trial_data / sigma**2
    quantiles.append(norm.cdf(truth, estimate, fisher_sd))
quantiles = np.array(quantiles)

probability = np.linspace(0, 1, 100)
band = 1.96 * np.sqrt(probability * (1 - probability) / n_trials)

fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.plot(probability, probability, "k--", lw=1)
ax.fill_between(
    probability,
    probability - band,
    probability + band,
    color="0.7",
    alpha=0.4,
    label="95% expected band",
)
for index, name in enumerate(["m", "c"]):
    fraction = (quantiles[:, index][None, :] < probability[:, None]).mean(axis=1)
    ax.plot(probability, fraction, label=name)
ax.set(
    xlabel="credible level",
    ylabel="fraction of truths inside",
    title=f"P-P plot, {n_trials} simulations",
    aspect="equal",
)
ax.legend()
plt.show()"""),
        md(
            """## Reference: the parameter-estimation checklist

Every analysis in the next two notebooks, and every published gravitational-wave
result, is built from exactly these pieces.

| Step | Question it answers | Where it can go wrong |
| --- | --- | --- |
| signal model $h(\\theta)$ | what could have produced the data? | waveform systematics, missing physics |
| noise model / PSD | what does "a good fit" mean quantitatively? | non-stationarity, lines, glitches, PSD uncertainty |
| likelihood $\\mathcal L(d\\mid\\theta)$ | how compatible are data and parameters? | wrong noise assumptions, correlated bins |
| prior $\\pi(\\theta)$ | what was allowed before these data? | unintended informativeness, hard boundaries |
| sampler | how do we explore the posterior? | poor tuning, unconverged chains, missed modes |
| diagnostics | can we trust this particular run? | too few effective samples, no burn-in check |
| evidence $\\mathcal Z$ | which model does the data prefer? | prior-volume dependence, under-converged runs |
| calibration (P-P) | is the whole pipeline correct? | only detectable over many simulations |

**Vocabulary quick reference**

- *Marginalisation* integrates a nuisance parameter out; *profiling* maximises
  over it. Section 3 showed these are different, and the LISA notebook measures
  exactly how different.
- *Burn-in* is the discarded start of a chain; *thinning* keeps every $k$-th
  sample. Thinning reduces storage, not Monte Carlo error.
- *Optimal SNR* assumes a perfect template; *matched-filter SNR* is what you
  actually recover from data. The LVK notebook computes both.
- *Credible interval* (Bayesian, probability over parameters) is not a
  *confidence interval* (frequentist, coverage over repeated experiments)."""
        ),
        md(
            """## Checks and takeaways

1. Widen the prior: which marginal changes most?
2. Halve the assumed `sigma`: does the posterior become more accurate or merely more confident?
3. Why may the PSD-dependent likelihood normalisation be dropped for fixed-PSD
   PE but not when comparing noise models?
4. Change the prior width in the evidence cell. Why does the posterior near its
   peak barely move while the Bayes factor can change?
5. Start the Metropolis chain at the true parameters. Does burn-in disappear,
   and is that a safe thing to do in general?
6. Reduce `n_live` in the nested sampler. What happens to `log Z`, and why is a
   single run's evidence not enough to quote an uncertainty?
7. In the Fisher cell, shrink `sigma` by a factor of ten. Why does the ellipse
   agree with the grid posterior even better?

**Takeaway:** PE is a model–data–noise calculation. A posterior is only as trustworthy as the waveform, response, PSD, priors, and computation that define it.

Adapted from the local `nz_bilby_cbc_workshop_2024` and its source, Colm Talbot's Bayesian inference tutorial."""
        ),
    ],
)


write(
    "01_lvk_compact_binary_parameter_estimation.ipynb",
    "LVK: compact-binary parameter estimation",
    [
        md(r"""## Goal and analysis map

Follow a compact version of the NZ Bilby workshop's full CBC flow:

\[
\theta_{\rm CBC}\rightarrow(h_+,h_\times)\rightarrow h_I
\rightarrow d_I=h_I+n_I\rightarrow\mathcal L_{\rm network}\rightarrow p(\theta\mid d).
\]

We use rippleGW for an actual IMRPhenomD waveform and Bilby for detector
geometry, PSDs, projection, and injection. Readable grid posteriors replace a
slow live sampler. The aim is to make every layer visible before a high-level
library combines them.

Section 3 adds the step that comes before any parameter estimation: **matched
filtering**, the search stage that finds the signal and produces the trigger.
Section 4 then estimates parameters from it, including a two-dimensional
posterior that shows the distance-inclination degeneracy behind
gravitational-wave distance uncertainties."""),
        code(
            """import os,sys,subprocess,importlib.util
IN_COLAB="COLAB_RELEASE_TAG" in os.environ
missing=[p for p in ("ripplegw","bilby") if importlib.util.find_spec(p) is None]
if missing:
    if IN_COLAB: subprocess.check_call([sys.executable,"-m","pip","install","-q","rippleGW==0.2.1","bilby==2.8.0"])
    else: raise ImportError("Install rippleGW==0.2.1 and bilby==2.8.0, or run in Colab.")"""
        ),
        code("""import logging
import numpy as np
import matplotlib.pyplot as plt
import bilby
from IPython.display import HTML,display
from matplotlib.animation import FuncAnimation
from jax import config
config.update("jax_enable_x64",True)
import jax.numpy as jnp
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
logging.getLogger("bilby").setLevel(logging.ERROR)
plt.style.use("seaborn-v0_8-whitegrid"); plt.rcParams["animation.html"]="jshtml"
rng=np.random.default_rng(20260817)"""),
        md(
            r"""## 1. CBC parameters

| Group | Examples | Main effect |
| --- | --- | --- |
| masses | $m_1,m_2$ or chirp mass $\mathcal M$ and $q=m_2/m_1$ | phase and merger frequency |
| spins | magnitudes and orientations | phase, precession, merger |
| matter/orbit | tides, eccentricity | extra phase and harmonics |
| location | right ascension, declination, distance | detector response and amplitude |
| orientation | inclination $\iota$, polarisation $\psi$, phase | relative polarisation content |
| time | geocentric coalescence time | detector arrival times |

\[
\mathcal M=\frac{(m_1m_2)^{3/5}}{(m_1+m_2)^{1/5}},\qquad
m_{\rm detector}=(1+z)m_{\rm source}.
\]

Intrinsic/extrinsic is useful bookkeeping, but parameters remain correlated in the posterior."""
        ),
        code("""sample_rate,duration,f_min=1024,4,20.; gps_time=1126259462.4
frequency=np.fft.rfftfreq(int(sample_rate*duration),1/sample_rate); mask=frequency>=f_min; df=frequency[1]-frequency[0]
def ripple_parameters(chirp_mass=None,m1=36.,m2=29.,chi1=.1,chi2=-.1,distance=800.,tc=0.,phase=.3,inclination=.5):
    mc,eta=ms_to_Mc_eta(jnp.array([m1,m2])); mc=mc if chirp_mass is None else chirp_mass
    return jnp.array([mc,eta,chi1,chi2,distance,tc,phase,inclination])
def polarizations(theta):
    hp,hx=gen_IMRPhenomD_hphc(jnp.asarray(frequency[mask]),theta,f_min)
    result={"plus":np.zeros(frequency.size,dtype=complex),"cross":np.zeros(frequency.size,dtype=complex)}
    result["plus"][mask]=np.asarray(hp); result["cross"][mask]=np.asarray(hx); return result
theta_true=ripple_parameters(); injection_polarizations=polarizations(theta_true)
print(f"Detector-frame chirp mass: {float(theta_true[0]):.3f} solar masses")"""),
        code(
            """fig,axes=plt.subplots(1,2,figsize=(11,3.4))
for name,h in injection_polarizations.items(): axes[0].loglog(frequency[mask],np.abs(h[mask]),label=name)
axes[0].set(xlabel="frequency [Hz]",ylabel="strain / Hz",title="Radiation has two polarisations"); axes[0].legend()
axes[1].plot(frequency[mask],np.unwrap(np.angle(injection_polarizations["plus"][mask])))
axes[1].set(xlabel="frequency [Hz]",ylabel="phase [rad]",title="Chirp mass is measured mainly through phase"); plt.show()"""
        ),
        md(
            r"""For a non-precessing circular binary, approximately
$h_+\propto(1+\cos^2\iota)/(2D_L)$ and $h_\times\propto\cos\iota/D_L$.
Inclination is the binary's orientation to us; polarisation angle rotates the plus/cross basis on the sky."""
        ),
        code(
            """frames=np.linspace(float(theta_true[0])-6,float(theta_true[0])+6,16)
fig,(aa,ap)=plt.subplots(1,2,figsize=(11,3.3)); la,=aa.loglog([],[]); lp,=ap.plot([],[])
aa.set(xlim=(20,512),ylim=(1e-25,3e-22),xlabel="frequency [Hz]",ylabel=r"$|h_+|$")
ap.set(xlim=(20,512),ylim=(-650,50),xlabel="frequency [Hz]",ylabel="relative phase [rad]")
def animate_mass(i):
    h=polarizations(theta_true.at[0].set(frames[i]))["plus"][mask]; phase=np.unwrap(np.angle(h)); phase-=phase[0]
    la.set_data(frequency[mask],np.abs(h)); lp.set_data(frequency[mask],phase); fig.suptitle(f"chirp mass = {frames[i]:.1f} solar masses"); return la,lp
animation=FuncAnimation(fig,animate_mass,frames=len(frames),interval=130); plt.close(fig); display(HTML(animation.to_jshtml()))"""
        ),
        md(
            """### Fast inspiral cartoon—physics intuition, not numerical relativity

This deliberately cheap animation connects orbital motion to a chirping quadrupole signal. It is not a surrogate or merger-remnant prediction: the actual waveform animation above is the quantitative one."""
        ),
        code("""cartoon_time=np.linspace(0,1,40); radius=1-.82*cartoon_time
orbital_phase=2*np.pi*(1.3*cartoon_time+5*cartoon_time**3)
x=radius*np.cos(orbital_phase); y=radius*np.sin(orbital_phase)
cartoon_strain=(1/radius)*np.cos(2*orbital_phase); cartoon_strain/=np.max(np.abs(cartoon_strain))
fig,(orbit_ax,strain_ax)=plt.subplots(1,2,figsize=(10,4))
body_1,=orbit_ax.plot([],[],"o",ms=9,color="C0"); body_2,=orbit_ax.plot([],[],"o",ms=7,color="C1"); separation,=orbit_ax.plot([],[],color="0.6")
strain_line,=strain_ax.plot([],[],color="C3"); marker,=strain_ax.plot([],[],"o",color="C3")
orbit_ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),aspect="equal",xlabel="x [cartoon]",ylabel="y [cartoon]",title="shrinking, accelerating orbit")
strain_ax.set(xlim=(0,1),ylim=(-1.1,1.1),xlabel="time to merger [cartoon]",ylabel="normalised strain",title="frequency and amplitude increase")
def animate_inspiral(i):
    body_1.set_data([x[i]],[y[i]]); body_2.set_data([-x[i]],[-y[i]]); separation.set_data([-x[i],x[i]],[-y[i],y[i]])
    strain_line.set_data(cartoon_time[:i+1],cartoon_strain[:i+1]); marker.set_data([cartoon_time[i]],[cartoon_strain[i]]); return body_1,body_2,separation,strain_line,marker
inspiral_animation=FuncAnimation(fig,animate_inspiral,frames=len(cartoon_time),interval=55)
plt.close(fig); display(HTML(inspiral_animation.to_jshtml()))"""),
        md(r"""## 2. From source to a detector network

For detector $I$,
\[
\tilde h_I=[F^I_+h_++F^I_\times h_\times]e^{-2\pi if\Delta t_I}.
\]
The antenna factors $F_+^I,F_\times^I$ depend on sky position, polarisation,
detector orientation, and sidereal time; $\Delta t_I$ is the arrival-time delay.
Bilby stores the detector geometry and PSDs and applies this projection.

If detector noises are independent conditional on their PSDs, the network
likelihood is a product, or equivalently a sum in log space:

\[
\log\mathcal L_{\rm net}(d\mid\theta)
=\sum_I\log\mathcal L_I(d_I\mid\theta)
=-\frac12\sum_I(d_I-h_I(\theta)\mid d_I-h_I(\theta))_I+C.
\]

The source parameters are shared across detectors; only the response and noise
weighting are detector-specific."""),
        code("""source_parameters=dict(ra=1.2,dec=-.4,psi=.7,geocent_time=gps_time)
ifos=bilby.gw.detector.InterferometerList(["H1","L1","V1"])
for ifo in ifos: ifo.set_strain_data_from_zero_noise(sampling_frequency=sample_rate,duration=duration,start_time=gps_time-2)
print("IFO     F+      Fx      delay [ms]")
for ifo in ifos:
    fp=ifo.antenna_response(source_parameters["ra"],source_parameters["dec"],gps_time,source_parameters["psi"],"plus")
    fx=ifo.antenna_response(source_parameters["ra"],source_parameters["dec"],gps_time,source_parameters["psi"],"cross")
    dt=ifo.time_delay_from_geocenter(source_parameters["ra"],source_parameters["dec"],gps_time)
    print(f"{ifo.name:>3}  {fp:+.3f}  {fx:+.3f}   {1e3*dt:+.2f}")"""),
        code("""fig,axes=plt.subplots(1,2,figsize=(11,3.4))
for ifo in ifos:
    asd=ifo.power_spectral_density.get_amplitude_spectral_density_array(frequency)
    axes[0].loglog(frequency[mask],asd[mask],label=ifo.name)
    response=ifo.get_detector_response(injection_polarizations,source_parameters,frequencies=frequency)
    axes[1].loglog(frequency[mask],np.abs(response[mask]),label=ifo.name)
axes[0].set(xlabel="frequency [Hz]",ylabel=r"ASD [1/$\\sqrt{\\mathrm{Hz}}$]",title="Each detector has a PSD")
axes[1].set(xlabel="frequency [Hz]",ylabel="projected strain / Hz",title="Each detector sees a different signal")
for ax in axes: ax.legend(); plt.show()"""),
        md(r"""### Fast animation: move the source across the detector network

At fixed sidereal time, changing right ascension moves the source around the
sky. The bars show each detector's root-sum-square antenna response
$\sqrt{F_+^2+F_\times^2}$. A real signal also carries phase, inclination,
polarisation, distance, and arrival-time information."""),
        code("""sky_ra_frames = np.linspace(-np.pi, np.pi, 24, endpoint=False)
detector_names = [ifo.name for ifo in ifos]

fig, (sky_ax, response_ax) = plt.subplots(
    1,
    2,
    figsize=(10, 3.8),
    subplot_kw={"projection": None},
)
sky_ax.remove()
sky_ax = fig.add_subplot(1, 2, 1, projection="mollweide")
source_marker, = sky_ax.plot([], [], "o", color="C3", ms=8)
sky_ax.grid(True)
sky_ax.set_title("source position")

bars = response_ax.bar(detector_names, np.zeros(len(ifos)))
response_ax.set(
    ylim=(0, 1),
    ylabel=r"$\\sqrt{F_+^2+F_\\times^2}$",
    title="instantaneous antenna response",
)

def animate_sky_response(frame):
    source_ra = sky_ra_frames[frame]
    source_marker.set_data([source_ra], [source_parameters["dec"]])
    for bar, ifo in zip(bars, ifos):
        f_plus = ifo.antenna_response(
            source_ra,
            source_parameters["dec"],
            gps_time,
            source_parameters["psi"],
            "plus",
        )
        f_cross = ifo.antenna_response(
            source_ra,
            source_parameters["dec"],
            gps_time,
            source_parameters["psi"],
            "cross",
        )
        bar.set_height(np.hypot(f_plus, f_cross))
    fig.suptitle(f"right ascension = {source_ra:+.2f} rad")
    return (source_marker, *bars)


response_animation = FuncAnimation(
    fig,
    animate_sky_response,
    frames=len(sky_ra_frames),
    interval=100,
)
plt.close(fig)
display(HTML(response_animation.to_jshtml()))"""),
        md(r"""## 3. Finding the signal first: matched filtering

Before anyone estimates parameters, something has to notice that a signal is
there. In real strain data a loud binary is still far below the noise: the
whitened signal peaks at a few tenths of the noise standard deviation, so no
amount of staring at the time series will show it.

The optimal linear filter for a *known* waveform in Gaussian noise is the
matched filter. Slide a normalised template through the data and record the
overlap as a function of trial coalescence time $\tau$:

\[
z(\tau)=\frac{(d\mid h_\tau)}{\sqrt{(h\mid h)}},\qquad
h_\tau(f)=h(f)\,e^{-2\pi i f\tau}.
\]

Because the time shift is only a phase ramp in frequency, the whole SNR time
series comes from a single inverse FFT rather than one integral per trial time.
Two numbers matter, and they are not the same:

- the **optimal SNR** $\rho_{\rm opt}=\sqrt{(h\mid h)}$, what a perfect template
  would achieve on average;
- the **matched-filter SNR**, the value actually recovered, which scatters about
  $\rho_{\rm opt}$ and is biased high at the peak because we maximised over
  $\tau$.

Searches repeat this over a bank of $\sim10^6$ templates. Parameter estimation
then starts from the resulting trigger."""),
        code("""bilby.core.utils.random.seed(2026)
noisy_ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for ifo in noisy_ifos:
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sample_rate, duration=duration, start_time=gps_time - 2
    )
    ifo.inject_signal_from_waveform_polarizations(
        source_parameters, injection_polarizations
    )

n_samples = int(sample_rate * duration)
segment_time = np.arange(n_samples) / sample_rate  # seconds after segment start
# fftshift puts zero trial offset in the middle: bilby already places the
# merger at geocent_time, so the peak should land at an offset of zero.
trial_offset = (np.arange(n_samples) - n_samples // 2) / sample_rate


def matched_filter(ifo, template_polarizations):
    \"\"\"Return the SNR time series and the optimal SNR for one detector.\"\"\"
    template = ifo.get_detector_response(
        template_polarizations, source_parameters, frequencies=frequency
    )
    psd = ifo.power_spectral_density_array
    usable = mask & np.isfinite(psd) & (psd > 0)
    integrand = np.zeros(frequency.size, dtype=complex)
    integrand[usable] = (
        ifo.frequency_domain_strain[usable] * np.conj(template[usable]) / psd[usable]
    )
    padded = np.zeros(n_samples, dtype=complex)
    padded[: integrand.size] = integrand
    z = 4 * df * n_samples * np.fft.ifft(padded)
    optimal = np.sqrt(4 * df * np.sum(np.abs(template[usable]) ** 2 / psd[usable]))
    return np.fft.fftshift(np.abs(z)) / optimal, optimal


fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ifo in noisy_ifos:
    snr_series, optimal_snr = matched_filter(ifo, injection_polarizations)
    peak = np.argmax(snr_series)
    axes[0].plot(trial_offset, snr_series, lw=0.7, label=ifo.name)
    axes[1].plot(trial_offset, snr_series, lw=1.2, label=ifo.name)
    print(
        f"{ifo.name}: optimal SNR {optimal_snr:5.2f} | "
        f"recovered peak {snr_series[peak]:5.2f} at "
        f"{trial_offset[peak]:+.4f} s"
    )
axes[0].set(
    xlabel="trial coalescence time offset [s]",
    ylabel=r"$|z(\\tau)|$",
    title="Matched-filter SNR across the whole segment",
)
axes[1].set(
    xlim=(-0.05, 0.05),
    xlabel="trial coalescence time offset [s]",
    title="Zoom: the trigger is sharply localised in time",
)
for ax in axes:
    ax.legend()
plt.show()"""),
        md(
            """### Animation: sliding the template through whitened data

Whitening divides each Fourier bin by the noise ASD so that every frequency
carries comparable noise, exactly the weighting the likelihood applies. The
left panel slides the whitened template across the whitened H1 data; the right
panel traces the SNR that the overlap produces. The signal is invisible by eye
in the data, yet the filter finds it because it adds the signal coherently over
hundreds of cycles while the noise adds incoherently."""
        ),
        code("""def whiten(frequency_series, psd):
    usable = mask & np.isfinite(psd) & (psd > 0)
    whitened = np.zeros(frequency.size, dtype=complex)
    whitened[usable] = frequency_series[usable] / np.sqrt(psd[usable] / (4 * df))
    return np.fft.irfft(whitened, n=n_samples)


h1 = noisy_ifos[0]
h1_psd = h1.power_spectral_density_array
whitened_data = whiten(h1.frequency_domain_strain, h1_psd)
whitened_template = whiten(
    h1.get_detector_response(
        injection_polarizations, source_parameters, frequencies=frequency
    ),
    h1_psd,
)
h1_snr_series, _ = matched_filter(h1, injection_polarizations)

lags = np.linspace(-0.3, 0.3, 45)
window = (segment_time > 1.3) & (segment_time < 2.35)

fig, (data_ax, snr_ax) = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=72)
data_ax.plot(segment_time[window], whitened_data[window], lw=0.6, color="0.55")
(template_line,) = data_ax.plot([], [], lw=1.4, color="C3")
data_ax.set(
    xlabel="time after segment start [s]",
    ylabel="whitened strain",
    title="whitened H1 data (grey) and trial template (red)",
)
(snr_trace,) = snr_ax.plot([], [], color="C0")
(snr_head,) = snr_ax.plot([], [], "o", color="C3")
snr_ax.set(
    xlim=(lags[0], lags[-1]),
    ylim=(0, 1.1 * h1_snr_series.max()),
    xlabel="template time shift [s]",
    ylabel=r"$|z(\\tau)|$",
    title="overlap accumulated by the filter",
)
fig.subplots_adjust(top=0.78, wspace=0.28)


def animate_filter(i):
    shift = int(round(lags[i] * sample_rate))
    shifted = np.roll(whitened_template, shift)
    template_line.set_data(segment_time[window], shifted[window])
    used = trial_offset <= lags[i]
    snr_trace.set_data(trial_offset[used], h1_snr_series[used])
    snr_value = np.interp(lags[i], trial_offset, h1_snr_series)
    snr_head.set_data([lags[i]], [snr_value])
    fig.suptitle(f"time shift {lags[i]:+.3f} s, SNR {snr_value:.1f}")
    return template_line, snr_trace, snr_head


filter_animation = FuncAnimation(
    fig, animate_filter, frames=len(lags), interval=110
)
plt.close(fig)
display(HTML(filter_animation.to_jshtml()))"""),
        md(
            """### A template only works if it is close enough

A search cannot use the true waveform, because the true parameters are what we
are trying to find. It uses a bank of templates and hopes one is close enough.
The cell below filters the same data with deliberately wrong chirp masses. The
recovered SNR falls away smoothly, and how fast it falls is exactly what sets
how densely a real template bank must be packed."""
        ),
        code("""mismatch_offsets = np.linspace(-4, 4, 25)
recovered_peaks = []
for offset in mismatch_offsets:
    trial = polarizations(theta_true.at[0].set(float(theta_true[0]) + offset))
    snr_series, _ = matched_filter(h1, trial)
    recovered_peaks.append(snr_series.max())

fig, ax = plt.subplots(figsize=(7.5, 3.3))
ax.plot(mismatch_offsets, recovered_peaks, "o-")
ax.axvline(0, color="k", ls="--", label="true chirp mass")
ax.set(
    xlabel="chirp-mass error of the template [solar masses]",
    ylabel="recovered peak SNR",
    title="Template mismatch loses signal-to-noise",
)
ax.legend()
plt.show()"""),
        md("""## 4. Inject and infer manually

We use zero-noise data so the demonstration is deterministic: the data equal
the injected signal, while the PSD still controls expected uncertainty. The
calculation below changes one shared source parameter, projects the waveform
into every detector, evaluates each detector's Whittle log likelihood, adds
them, applies a prior, and normalises the posterior. Replace
`set_strain_data_from_zero_noise` with Bilby's PSD-noise method to study repeated
noise realisations."""),
        code(
            """for ifo in ifos: ifo.inject_signal_from_waveform_polarizations(source_parameters,injection_polarizations)
print("Network optimal SNR:",round(np.sqrt(sum(ifo.meta_data["optimal_SNR"]**2 for ifo in ifos)),2))

def detector_log_likelihood(ifo,model_polarizations):
    model=ifo.get_detector_response(model_polarizations,source_parameters,frequencies=frequency)
    residual=ifo.frequency_domain_strain-model
    psd=ifo.power_spectral_density_array
    return -2*df*np.sum(np.abs(residual[mask])**2/psd[mask])

mass_grid=np.linspace(float(theta_true[0])-2,float(theta_true[0])+2,141)
logL_by_ifo={ifo.name:[] for ifo in ifos}
for mc in mass_grid:
    model=polarizations(theta_true.at[0].set(mc))
    for ifo in ifos: logL_by_ifo[ifo.name].append(detector_log_likelihood(ifo,model))
logL_network=np.sum([logL_by_ifo[name] for name in logL_by_ifo],axis=0)
def density(logp):
    p=np.exp(logp-np.max(logp)); return p/np.trapezoid(p,mass_grid)
log_prior_mass=np.where((mass_grid>=mass_grid[0])&(mass_grid<=mass_grid[-1]),0.,-np.inf)
posterior_h1=density(np.array(logL_by_ifo["H1"])+log_prior_mass)
posterior_network=density(logL_network+log_prior_mass)
fig,ax=plt.subplots(figsize=(8,3.4)); ax.plot(mass_grid,posterior_h1,label="H1 only")
ax.plot(mass_grid,posterior_network,label="H1+L1+V1"); ax.axvline(float(theta_true[0]),color="k",ls="--",label="injection")
ax.set(xlabel="detector-frame chirp mass [solar masses]",ylabel="posterior density",title="A coherent network gives more information"); ax.legend(); plt.show()"""
        ),
        md("""### Put the same likelihood behind Bilby's interface

Bilby does not require the likelihood to be a black box. A likelihood class
declares the sampled parameters and a `log_likelihood` method. Here Bilby wraps
the exact network calculation above; the assertion checks that the library
interface and the manual values agree."""),
        code("""class ChirpMassLikelihood(bilby.Likelihood):
    def __init__(self):
        super().__init__()

    def log_likelihood(self, parameters=None):
        trial_theta = theta_true.at[0].set(parameters["chirp_mass"])
        trial_polarizations = polarizations(trial_theta)
        return sum(
            detector_log_likelihood(ifo, trial_polarizations) for ifo in ifos
        )


bilby_likelihood = ChirpMassLikelihood()
bilby_priors = {
    "chirp_mass": bilby.core.prior.Uniform(
        minimum=mass_grid[0],
        maximum=mass_grid[-1],
        name="chirp_mass",
        unit="solar masses",
    )
}

bilby_log_likelihood = []
for chirp_mass in mass_grid:
    bilby_log_likelihood.append(
        bilby_likelihood.log_likelihood(parameters={"chirp_mass": chirp_mass})
    )

np.testing.assert_allclose(bilby_log_likelihood, logL_network)
print("Bilby likelihood agrees with the manual network calculation.")
print("Prior:", bilby_priors["chirp_mass"])"""),
        md(r"""### A two-dimensional posterior with a real degeneracy

A one-parameter scan hides the feature that dominates real CBC results:
parameters are correlated, and some are correlated so strongly that they are
effectively measured only in combination.

The classic example is distance and inclination. For the dominant quadrupole
mode of a circular binary,

\[
h_+\propto\frac{1+\cos^2\iota}{2D_L},\qquad
h_\times\propto\frac{\cos\iota}{D_L},
\]

so both parameters enter only as amplitudes. Moving the source further away and
tilting it face-on both make the signal louder or quieter in nearly the same
way. This is why gravitational-wave distances are much less precise than
chirp masses, and why standard-siren cosmology cares so much about breaking it.

Because inclination and distance affect IMRPhenomD only through these
prefactors, we can rescale the injected polarisations instead of regenerating
the waveform, which makes a two-dimensional grid cheap. The cell asserts that
this shortcut reproduces rippleGW exactly.

Two caveats worth carrying forward. This grid uses a **flat prior on distance**
for simplicity; a real analysis uses a uniform-in-comoving-volume prior, which
grows like $D_L^2$ and therefore pushes the posterior towards larger distances.
And the posterior is one-sided in inclination here because we restricted
$\iota\le\pi/2$; the full problem is also nearly symmetric under
$\iota\rightarrow\pi-\iota$, giving the familiar two-lobed structure."""),
        code("""true_distance, true_inclination = 800.0, 0.5


def scaled_polarizations(distance, inclination):
    \"\"\"Rescale the injection to a new distance and inclination.\"\"\"
    plus_ratio = ((1 + np.cos(inclination) ** 2) / 2) / (
        (1 + np.cos(true_inclination) ** 2) / 2
    )
    cross_ratio = np.cos(inclination) / np.cos(true_inclination)
    distance_ratio = true_distance / distance
    return {
        "plus": injection_polarizations["plus"] * plus_ratio * distance_ratio,
        "cross": injection_polarizations["cross"] * cross_ratio * distance_ratio,
    }


# The shortcut must agree with a full rippleGW call.
check = polarizations(ripple_parameters(distance=1300.0, inclination=0.9))
shortcut = scaled_polarizations(1300.0, 0.9)
for polarisation in ("plus", "cross"):
    np.testing.assert_allclose(
        shortcut[polarisation][mask], check[polarisation][mask], rtol=1e-10
    )
print("Amplitude rescaling reproduces rippleGW to machine precision.")

distance_grid = np.linspace(450, 1250, 70)
inclination_grid = np.linspace(0.02, np.pi / 2 - 0.02, 66)
logL_grid = np.array(
    [
        [
            sum(
                detector_log_likelihood(ifo, scaled_polarizations(distance, inclination))
                for ifo in ifos
            )
            for inclination in inclination_grid
        ]
        for distance in distance_grid
    ]
)

joint_posterior = np.exp(logL_grid - logL_grid.max())
joint_posterior /= np.trapezoid(
    np.trapezoid(joint_posterior, inclination_grid, axis=1), distance_grid
)
distance_marginal = np.trapezoid(joint_posterior, inclination_grid, axis=1)
inclination_marginal = np.trapezoid(joint_posterior, distance_grid, axis=0)"""),
        code("""fig = plt.figure(figsize=(9, 5.5))
grid_spec = fig.add_gridspec(
    2, 2, width_ratios=(4, 1.4), height_ratios=(1.4, 4), wspace=0.05, hspace=0.05
)
joint_ax = fig.add_subplot(grid_spec[1, 0])
top_ax = fig.add_subplot(grid_spec[0, 0], sharex=joint_ax)
side_ax = fig.add_subplot(grid_spec[1, 1], sharey=joint_ax)

joint_ax.contourf(
    distance_grid, inclination_grid, joint_posterior.T, levels=20, cmap="magma"
)
joint_ax.plot(true_distance, true_inclination, "c*", ms=14, label="injection")
joint_ax.set(
    xlabel="luminosity distance [Mpc]", ylabel=r"inclination $\\iota$ [rad]"
)
joint_ax.legend(loc="upper right", facecolor="white", framealpha=0.9)

top_ax.plot(distance_grid, distance_marginal, color="C0")
top_ax.axvline(true_distance, color="k", ls="--")
top_ax.set_ylabel("marginal")
top_ax.tick_params(labelbottom=False)

side_ax.plot(inclination_marginal, inclination_grid, color="C0")
side_ax.axhline(true_inclination, color="k", ls="--")
side_ax.set_xlabel("marginal")
side_ax.tick_params(labelleft=False)
fig.suptitle("The distance-inclination degeneracy")
plt.show()


def credible_interval(grid, density, probability=0.9):
    cdf = np.r_[0, np.cumsum((density[:-1] + density[1:]) * np.diff(grid) / 2)]
    cdf /= cdf[-1]
    tail = (1 - probability) / 2
    return np.interp([tail, 0.5, 1 - tail], cdf, grid)


low, median, high = credible_interval(distance_grid, distance_marginal)
print(f"injected distance : {true_distance:.0f} Mpc")
print(f"posterior median  : {median:.0f} Mpc")
print(f"90% interval      : [{low:.0f}, {high:.0f}] Mpc")
print(
    "Fractional distance precision: "
    f"{(high - low) / (2 * median):.0%}, far worse than the chirp mass."
)"""),
        md(
            """## 5. Why a network localises the sky

Timing alone gives a ring with two sites and smaller regions with three. Real Bilby localisation also uses coherent phase, antenna amplitudes, polarisation, distance–inclination correlations, waveform uncertainty, and sky priors."""
        ),
        code(
            """ra=np.linspace(-np.pi,np.pi,91); dec=np.linspace(-np.pi/2,np.pi/2,46); RA,DEC=np.meshgrid(ra,dec)
delays={ifo.name:np.array([[ifo.time_delay_from_geocenter(r,d,gps_time) for r in ra] for d in dec]) for ifo in ifos}
observed={ifo.name:ifo.time_delay_from_geocenter(source_parameters["ra"],source_parameters["dec"],gps_time) for ifo in ifos}; sigma_t=3e-4
def timing_likelihood(names):
    ref=names[0]; value=np.zeros_like(RA)
    for name in names[1:]: value-=.5*((delays[name]-delays[ref]-(observed[name]-observed[ref]))/sigma_t)**2
    return value
fig,axes=plt.subplots(1,2,figsize=(12,4),subplot_kw={"projection":"mollweide"})
for ax,names,title in zip(axes,[["H1","L1"],["H1","L1","V1"]],["two detectors: delay ring","three detectors: smaller regions"]):
    ll=timing_likelihood(names); sky=np.exp(ll-ll.max()); ax.contourf(RA,DEC,sky,levels=np.linspace(.05,1,15),cmap="magma")
    ax.plot(source_parameters["ra"],source_parameters["dec"],"c*",ms=10); ax.set_title(title); ax.grid(True)
plt.show()"""
        ),
        md(
            r"""## 6. From events to a population

Event-level posteriors become inputs to hierarchical inference. If $\Lambda$ describes a population,
\[
p(\Lambda\mid\{d_i\},\mathrm{det})\propto p(\Lambda)
\prod_i\frac{\int p(d_i\mid\theta)p(\theta\mid\Lambda)d\theta}{\alpha(\Lambda)}.
\]
$\alpha(\Lambda)$ is the detectable fraction. Ignoring it confuses the observed catalogue with the astrophysical population."""
        ),
        code(
            """from scipy.stats import norm
population_mean,population_width=28.,5.; all_masses=rng.normal(population_mean,population_width,8000)
all_masses=all_masses[(all_masses>8)&(all_masses<55)]
def detection_probability(mass): return 1/(1+np.exp(-(mass-22)/3.5))
detected=all_masses[rng.random(all_masses.size)<detection_probability(all_masses)][:40]
mean_grid=np.linspace(18,38,320); integration_grid=np.linspace(8,55,900); naive=[]; corrected=[]
for mean in mean_grid:
    event_term=norm.logpdf(detected,mean,population_width).sum()
    alpha=np.trapezoid(norm.pdf(integration_grid,mean,population_width)*detection_probability(integration_grid),integration_grid)
    naive.append(event_term); corrected.append(event_term-len(detected)*np.log(alpha))
def normalise_population(logp):
    p=np.exp(logp-np.max(logp)); return p/np.trapezoid(p,mean_grid)
fig,axes=plt.subplots(1,2,figsize=(11,3.4)); mass_axis=np.linspace(8,55,300)
axes[0].hist(all_masses,bins=35,density=True,histtype="step",label="underlying"); axes[0].hist(detected,bins=13,density=True,alpha=.5,label="detected")
axes[0].plot(mass_axis,detection_probability(mass_axis)/20,"--",label="selection (scaled)"); axes[0].set(xlabel="mass [toy units]",ylabel="density",title="Detected is not underlying"); axes[0].legend()
axes[1].plot(mean_grid,normalise_population(np.array(naive)),label="ignores selection"); axes[1].plot(mean_grid,normalise_population(np.array(corrected)),label="selection-aware")
axes[1].axvline(population_mean,color="k",ls="--",label="injection"); axes[1].set(xlabel="population mean",ylabel="posterior density",title="Selection changes the answer"); axes[1].legend(); plt.show()"""
        ),
        code("""print(f"Injected population mean: {population_mean:.2f}")
print(f"Detected-catalogue mean: {detected.mean():.2f}")
print(f"Naive MAP: {mean_grid[np.argmax(naive)]:.2f}")
print(f"Selection-aware MAP: {mean_grid[np.argmax(corrected)]:.2f}")"""),
        md(
            """This compact example treats masses as exactly measured. Real population inference reweights uncertain event posteriors, estimates selection with injection campaigns, infers several hyperparameters and often the rate, and checks sensitivity to event-level priors and waveform systematics.

## 7. Bilby production composition

```python
waveform_generator = bilby.gw.WaveformGenerator(...)
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator)
result = bilby.run_sampler(likelihood, priors, sampler="dynesty", ...)
```

Bilby then handles response projection, detector likelihoods, common parameters,
priors, marginalisations, sampling, and result objects. For this live notebook,
the grid calculation is the sampler: it is exact for the displayed
one-dimensional discretisation and finishes quickly. A production run replaces
the toy likelihood wrapper with a waveform generator and a stochastic sampler.

## Boundary and extensions

- The live posterior frees only chirp mass; production BBH analyses may have about fifteen parameters plus nuisance/systematic choices.
- Design PSDs and zero noise are pedagogical. Real data contain PSD uncertainty, lines, glitches, non-stationarity, and calibration uncertainty.
- Use the NZ workshop's GW150914 section or [GWOSC tutorials](https://gwosc.org/tutorials/) as a real-data follow-up.
- Extend the mock catalogue by giving each event a mass posterior instead of an exact mass.

Adapted substantially from `nz_bilby_cbc_workshop_2024`, with its injection → PSD → prior → likelihood → result structure."""
        ),
    ],
)


write(
    "02_lisa_parameter_estimation_and_global_fit.ipynb",
    "LISA: sensitivity, response, and the global fit",
    [
        md(r"""## Goal and analysis map

This notebook follows the LISA Analysis Tools Workshop progression:

\[
\text{sensitivity}\rightarrow\text{TDI data}\rightarrow(a\mid b)
\rightarrow\mathrm{SNR}\rightarrow\mathcal L\rightarrow
\text{single source}\rightarrow\text{unknown overlapping catalogue}.
\]

We use `lisatools` for LISA sensitivity curves and its `AnalysisContainer`, and
JaxGB for an actual moving-constellation Galactic-binary response. You will
first calculate a frequency likelihood manually, then verify the same objects
through the LISA Analysis Tools interface. The final exercise is a miniature
version of the LATW global-fit challenge."""),
        code(
            """import os,sys,subprocess,importlib.util
IN_COLAB="COLAB_RELEASE_TAG" in os.environ
needed=("lisatools","gpubackendtools","jaxgb","eryn")
if any(importlib.util.find_spec(package) is None for package in needed):
    if IN_COLAB:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","lisaanalysistools==1.2.5","gpubackendtools==0.1.1","jaxgb==0.2.1","astropy==7.2.0","eryn==1.2.6"])
    else: raise ImportError("Install the pinned LISA requirements, or run in Colab.")"""
        ),
        code(
            '''import itertools
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML,display
from matplotlib.animation import FuncAnimation
from jax import config
config.update("jax_enable_x64",True)
warnings.filterwarnings("ignore", message="IProgress not found.*")
from lisatools.sensitivity import A1TDISens,E1TDISens,SensitivityMatrix,get_sensitivity
from lisatools.utils.constants import YRSID_SI
from lisaorbits import EqualArmlengthOrbits
from jaxgb.jaxgb import JaxGB
from jaxgb.params import GBObject
rng=np.random.default_rng(20260817); plt.style.use("seaborn-v0_8-whitegrid"); plt.rcParams["animation.html"]="jshtml"'''
        ),
        md(
            r"""## 1. LISA's band and source zoo

Ground-based detectors observe roughly tens of Hz to kHz. LISA targets approximately $10^{-4}$–$10^{-1}$ Hz, containing Galactic compact binaries, massive-black-hole binaries, EMRIs, stellar-origin binaries, stochastic backgrounds, and instrument noise. Long observations make many signals overlap.

Unlike a static right-angle detector, LISA is a heliocentric triangle that cartwheels as it orbits. Six delayed one-way laser links are combined into time-delay interferometry (TDI) variables. Orbital modulation helps localisation, while finite arms create a frequency-dependent response."""
        ),
        md("""### Why LISA parameter estimation is unusually coupled

| Feature | Typical transient LVK CBC analysis | LISA analysis |
| --- | --- | --- |
| signal duration in band | seconds to minutes for many CBCs | months to years for many sources |
| response during one signal | detector geometry often changes little | constellation motion modulates the signal |
| data channels | separated ground detectors | correlated laser links combined into TDI |
| source overlap | often analyse a short segment around one event | many persistent sources share the same bins |
| noise/foreground | PSD estimated around an event, with caveats | instrument noise and astrophysical foreground may evolve together |
| catalogue size | event trigger supplies a candidate | number of resolvable sources can be unknown |

The hard part is not that Bayes' theorem changes. The signal, response, noise,
and catalogue blocks become more strongly coupled, so fitting one source while
treating everything else as fixed can bias the residual seen by the next
source."""),
        code(
            """year=YRSID_SI; AU=149597870700.; orbits=EqualArmlengthOrbits(); times=np.linspace(0,year,240)
positions=np.asarray(orbits.compute_position(times,[1,2,3])); fig,ax=plt.subplots(figsize=(5.4,5.4))
for i,label in enumerate(["spacecraft 1","spacecraft 2","spacecraft 3"]): ax.plot(positions[:,i,0]/AU,positions[:,i,1]/AU,label=label)
ax.plot(0,0,"o",color="gold",mec="k",label="Sun"); ax.set(xlabel="heliocentric x [AU]",ylabel="heliocentric y [AU]",title="An explicit LISA orbit model",aspect="equal"); ax.legend(); plt.show()"""
        ),
        md(
            """### Fast animation: the moving constellation

The orbital motion is not decorative: it amplitude-, phase-, and frequency-modulates long-lived signals and encodes sky position. This animation uses precomputed orbit coordinates, so it remains fast in Colab."""
        ),
        code(
            """orbit_frames=np.arange(0,len(times),10); fig,ax=plt.subplots(figsize=(5.2,5.2)); triangle,=ax.plot([],[],"o-",lw=1.5); trail_lines=[ax.plot([],[],alpha=.35)[0] for _ in range(3)]
ax.plot(0,0,"o",color="gold",mec="k"); ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),aspect="equal",xlabel="x [AU]",ylabel="y [AU]")
def animate_constellation(frame):
    i=orbit_frames[frame]; current=positions[i,:,:2]/AU; closed=np.vstack([current,current[0]])
    triangle.set_data(closed[:,0],closed[:,1])
    for spacecraft,line in enumerate(trail_lines): line.set_data(positions[:i+1,spacecraft,0]/AU,positions[:i+1,spacecraft,1]/AU)
    ax.set_title(f"LISA constellation: day {times[i]/86400:.0f}"); return (triangle,*trail_lines)
orbit_animation=FuncAnimation(fig,animate_constellation,frames=len(orbit_frames),interval=90)
plt.close(fig); display(HTML(orbit_animation.to_jshtml()))"""
        ),
        md(
            """## 2. Sensitivity and Galactic confusion

As in LATW Tutorial 1, start with the noise model. The unresolved Galactic foreground changes with observing time because longer data resolve and subtract more binaries."""
        ),
        code(
            """f_curve=np.logspace(-5,-1,1800)
instrument=SensitivityMatrix(f_curve,[A1TDISens,E1TDISens])
one_year=SensitivityMatrix(f_curve,[A1TDISens,E1TDISens],stochastic_params=(1*year,))
four_year=SensitivityMatrix(f_curve,[A1TDISens,E1TDISens],stochastic_params=(4*year,))
fig,ax=plt.subplots(figsize=(8,3.6))
ax.loglog(f_curve,np.sqrt(instrument.sens_mat[0]),label="instrument only")
ax.loglog(f_curve,np.sqrt(one_year.sens_mat[0]),label="+ 1-year Galactic foreground")
ax.loglog(f_curve,np.sqrt(four_year.sens_mat[0]),label="+ 4-year Galactic foreground")
ax.set(xlabel="frequency [Hz]",ylabel=r"TDI A ASD [1/$\\sqrt{\\mathrm{Hz}}$]",title="Sensitivity is part of the likelihood"); ax.legend(); plt.show()"""
        ),
        md(
            """## 3. Real LISA data will be more complicated

| Complication | What students should ask | Analysis consequence |
| --- | --- | --- |
| data gaps and irregular availability | why is data missing; is the gap informative? | Fourier bins become coupled; windowing/inpainting/gap-aware likelihoods |
| time-varying instrument noise | is one mission-long PSD meaningful? | segmented, time-frequency, or explicitly non-stationary noise models |
| changing Galactic foreground | which binaries become resolvable with time? | foreground and catalogue must be inferred together |
| moving/breathing constellation | are delays and orbits known accurately enough? | time-dependent link response and TDI generation |
| glitches, clock/laser artefacts, calibration | can an instrumental transient mimic a source? | extra nuisance models, vetoes, robust likelihoods |
| overlapping source classes | what belongs to the residual? | global rather than source-by-source inference |
| unknown catalogue size | how many binaries are present? | trans-dimensional/RJ methods and label-switching care |

The following lightweight laboratory is intentionally editable. Change the gap, drift, and noise-growth parameters and rerun it."""
        ),
        code("""from scipy.signal import welch,spectrogram
# Student playground: all three controls are deliberately visible.
GAP_DAYS=(11,14); FREQUENCY_DRIFT=2e-11; NOISE_GROWTH=.9
cadence=60.; mission_days=28.; toy_time=np.arange(0,mission_days*86400,cadence); sample_rate_toy=1/cadence
noise_scale=1+NOISE_GROWTH*toy_time/toy_time[-1]
phase=2*np.pi*(3e-3*toy_time+.5*FREQUENCY_DRIFT*toy_time**2)
continuous_data=noise_scale*rng.normal(size=toy_time.size)+1.5*np.sin(phase)
available=~((toy_time>=GAP_DAYS[0]*86400)&(toy_time<GAP_DAYS[1]*86400))
gapped_data=continuous_data.copy(); gapped_data[~available]=0  # zero fill only to expose leakage below

early=continuous_data[toy_time<7*86400]; late=continuous_data[toy_time>21*86400]
f_early,p_early=welch(early,fs=sample_rate_toy,nperseg=4096)
f_late,p_late=welch(late,fs=sample_rate_toy,nperseg=4096)
window=np.hanning(toy_time.size); fft_frequency=np.fft.rfftfreq(toy_time.size,cadence)
fft_full=np.abs(np.fft.rfft(window*continuous_data)); fft_gap=np.abs(np.fft.rfft(window*gapped_data))
f_spec,t_spec,p_spec=spectrogram(gapped_data,fs=sample_rate_toy,nperseg=2048,noverlap=1536)

fig,axes=plt.subplots(2,2,figsize=(12,7))
axes[0,0].plot(toy_time[::80]/86400,available[::80].astype(int)); axes[0,0].set(xlabel="mission time [days]",ylabel="available?",title="A three-day gap")
axes[0,1].loglog(f_early[1:],np.sqrt(p_early[1:]),label="week 1"); axes[0,1].loglog(f_late[1:],np.sqrt(p_late[1:]),label="week 4")
axes[0,1].set(xlabel="frequency [Hz]",ylabel="ASD [toy]",title="Noise level changes with time"); axes[0,1].legend()
near=(fft_frequency>2.7e-3)&(fft_frequency<3.4e-3); axes[1,0].semilogy(1e3*fft_frequency[near],fft_full[near],label="continuous")
axes[1,0].semilogy(1e3*fft_frequency[near],fft_gap[near],label="gap zero-filled",alpha=.8); axes[1,0].set(xlabel="frequency [mHz]",ylabel="FFT magnitude",title="A gap spreads power across bins"); axes[1,0].legend()
band=(f_spec>2.7e-3)&(f_spec<3.4e-3); image=axes[1,1].pcolormesh(t_spec/86400,1e3*f_spec[band],np.log10(p_spec[band]+1e-30),shading="auto")
axes[1,1].set(xlabel="mission time [days]",ylabel="frequency [mHz]",title="Drifting line + gap in time–frequency"); fig.colorbar(image,ax=axes[1,1],label="log power")
fig.tight_layout(); plt.show()"""),
        md(
            """**Do not interpret zero-filling as the recommended gap treatment.** It is used here because its spectral leakage is immediately visible. A research analysis must define how gaps, edges, non-stationarity, and missing-data uncertainty enter the likelihood.

### Suggested investigations

- Set `GAP_DAYS=(0, 0)` and verify that the leakage pattern changes.
- Increase `FREQUENCY_DRIFT`; when does a stationary single-bin model fail?
- Fit the early PSD to late data. Which likelihood assumption is violated?
- Replace the hard gap with a taper and compare leakage against loss of usable data.
- Add a short glitch inside versus outside the gap."""
        ),
        md(
            r"""## 4. Inner product, SNR, and likelihood

For independent A and E channels,
\[
(a\mid b)=4\Delta f\,\mathrm{Re}\sum_{X\in\{A,E\},k}\frac{a_{Xk}^*b_{Xk}}{S_X(f_k)},
\quad\rho_{\rm opt}=\sqrt{(h\mid h)},
\quad\log\mathcal L=-\tfrac12(d-h\mid d-h).
\]

These are the same objects as in LVK analysis. What changes is the instrument response, source durations, channels, band, and global model."""
        ),
        code(
            """t_obs=90*86400.; simulator=JaxGB(orbits,t_obs=t_obs,t0=0,n=128)
source=GBObject(f0=np.array([3e-3]),fdot=np.array([1e-17]),A=np.array([2e-22]),ra=np.array([1.]),dec=np.array([.4]),psi=np.array([.3]),iota=np.array([.8]),phi0=np.array([.2]),t_init=0.)
parameters=source.to_jaxgb_array(t0=0); A,E,T=simulator.get_tdi(parameters,tdi_generation=2,tdi_combination="AET")
frequency=np.asarray(simulator.get_frequency_grid(simulator.get_kmin(parameters[:,0])))[0]; df=1/t_obs
template=np.stack([np.asarray(A)[0],np.asarray(E)[0]])
psd=np.stack([get_sensitivity(frequency,sens_fn=A1TDISens,stochastic_params=(t_obs,)),get_sensitivity(frequency,sens_fn=E1TDISens,stochastic_params=(t_obs,))])
def inner(a,b): return 4*df*np.real(np.sum(np.conj(a)*b/psd))
optimal_snr=np.sqrt(inner(template,template))
print(f"90-day optimal A+E SNR: {optimal_snr:.2f}")
fig,axes=plt.subplots(1,2,figsize=(11,3.4))
axes[0].plot(1e3*frequency,np.abs(template[0]),label="A"); axes[0].plot(1e3*frequency,np.abs(template[1]),label="E")
axes[0].set(xlabel="frequency [mHz]",ylabel="response magnitude",title="JaxGB second-generation TDI"); axes[0].legend()
axes[1].semilogy(1e3*frequency,4*df*np.sum(np.abs(template)**2/psd,axis=0))
axes[1].set(xlabel="frequency [mHz]",ylabel=r"contribution to $\\rho^2$",title="PSD-weighted information by bin"); plt.show()"""
        ),
        md(r"""### Manual one-parameter likelihood

Perturb the source frequency, regenerate the moving-constellation response, and
compare optimal SNR with detected/matched SNR. A loud template can still match
the data poorly.

Watch the two scales in the plots below. The overlap between two templates
decays once they are separated by about one frequency bin, $1/T_{\rm obs}$.
The *likelihood* is narrower than that by roughly the signal-to-noise ratio,
which is why a 90-day observation pins $f_0$ to a small fraction of a bin.
Longer missions help twice over: more bins, and more SNR per source."""),
        code(
            '''def trial_template(f0_offset):
    """Regenerate the moving-constellation response at a shifted frequency."""
    trial = GBObject(
        f0=np.array([3e-3 + f0_offset]),
        fdot=np.array([1e-17]),
        A=np.array([2e-22]),
        ra=np.array([1.0]),
        dec=np.array([0.4]),
        psi=np.array([0.3]),
        iota=np.array([0.8]),
        phi0=np.array([0.2]),
        t_init=0.0,
    )
    kmin = int(simulator.get_kmin(parameters[:, 0])[0])
    a, e, _ = simulator.sum_tdi(
        trial.to_jaxgb_array(t0=0),
        kmin,
        kmin + simulator.n,
        tdi_generation=2,
        tdi_combination="AET",
    )
    return np.stack([np.asarray(a), np.asarray(e)])


# Two very different scales matter here. The overlap between templates decays
# over roughly a frequency bin, 1/T_obs, but the likelihood is narrower than
# that by about the signal-to-noise ratio.
wide_offsets = np.linspace(-7e-7, 7e-7, 61)
detected = []
for offset in wide_offsets:
    h = trial_template(offset)
    detected.append(inner(template, h) / np.sqrt(inner(h, h)))

offsets = np.linspace(-2e-8, 2e-8, 61)
logL, trial_templates = [], []
for offset in offsets:
    h = trial_template(offset)
    trial_templates.append(h)
    logL.append(-0.5 * inner(template - h, template - h))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(1e9 * wide_offsets, detected)
axes[0].axhline(optimal_snr, color="k", ls="--", label="optimal SNR")
axes[0].axvline(1e9 / t_obs, color="C3", ls=":", label=r"one bin, $1/T_{\\rm obs}$")
axes[0].set(
    xlabel="frequency offset [nHz]",
    ylabel="detected SNR",
    title="Match falls away over about one bin",
)
axes[0].legend(fontsize=8)
axes[1].plot(1e9 * offsets, np.array(logL) - np.max(logL))
axes[1].axhline(-0.5, color="0.6", ls=":")
axes[1].set(
    xlabel="frequency offset [nHz]",
    ylabel=r"$\\Delta \\log \\mathcal{L}$",
    title="The likelihood is far narrower still",
)
plt.show()

print(f"one frequency bin      : {1 / t_obs:.3e} Hz")
print(f"plotted likelihood span: {offsets[-1] - offsets[0]:.3e} Hz")'''
        ),
        md(r"""### From a likelihood scan to a forecast: the Fisher matrix

The scan above varied one parameter and held the rest fixed. That is a
**conditional** slice, not a posterior width. To forecast what LISA measures,
the standard tool is the Fisher matrix,

\[
F_{ij}=\left(\frac{\partial h}{\partial\theta_i}\Big|
\frac{\partial h}{\partial\theta_j}\right),\qquad
\sigma_i^{\rm marginal}=\sqrt{(F^{-1})_{ii}},\qquad
\sigma_i^{\rm conditional}=1/\sqrt{F_{ii}}.
\]

We build it by finite-differencing the *full moving-constellation TDI
response*, so the derivatives include the orbital modulation. Two checks make
this concrete:

- amplitude is a pure scaling, so $\sigma_{\ln A}$ must equal $1/\rho$ exactly;
- the frequency scan from the previous cell must reproduce the **conditional**
  error, which is smaller than the marginal one by $\sqrt{1-\rho_{f_0\phi_0}^2}$.

That second point is the LISA-scale version of the lesson from the basics
notebook: marginalising over a correlated parameter is not the same as fixing
it, and quoting a conditional error as if it were a measurement uncertainty
understates it. Fisher forecasts are cheap and ubiquitous, but they assume high
SNR and near-linearity; at low SNR the real posterior is not an ellipse."""),
        code('''fisher_truth = dict(
    f0=3e-3, fdot=1e-17, A=2e-22, ra=1.0, dec=0.4, psi=0.3, iota=0.8, phi0=0.2
)


def fisher_response(**overrides):
    """TDI A and E on the fixed band, for the source with parameters replaced."""
    values = dict(fisher_truth)
    values.update(overrides)
    array = GBObject(
        **{key: np.array([value]) for key, value in values.items()}, t_init=0.0
    ).to_jaxgb_array(t0=0)
    a, e, _ = simulator.sum_tdi(
        array,
        int(simulator.get_kmin(parameters[:, 0])[0]),
        int(simulator.get_kmin(parameters[:, 0])[0]) + simulator.n,
        tdi_generation=2,
        tdi_combination="AET",
    )
    return np.stack([np.asarray(a), np.asarray(e)])


def derivative(key, step):
    """Central difference; amplitude is differentiated with respect to log A."""
    if key == "A":
        plus = fisher_response(A=fisher_truth["A"] * np.exp(step))
        minus = fisher_response(A=fisher_truth["A"] * np.exp(-step))
    else:
        plus = fisher_response(**{key: fisher_truth[key] + step})
        minus = fisher_response(**{key: fisher_truth[key] - step})
    return (plus - minus) / (2 * step)


fisher_labels = [r"$f_0$", r"$\\ln A$", r"$\\phi_0$"]
derivatives = [derivative("f0", 1e-9), derivative("A", 1e-3), derivative("phi0", 1e-3)]
fisher_matrix = np.array([[inner(a, b) for b in derivatives] for a in derivatives])
fisher_covariance = np.linalg.inv(fisher_matrix)

marginal_sd = np.sqrt(np.diag(fisher_covariance))
conditional_sd = 1 / np.sqrt(np.diag(fisher_matrix))
correlation = fisher_covariance / np.outer(marginal_sd, marginal_sd)

print(f"optimal SNR                 : {optimal_snr:.3f}")
print(f"sigma(ln A) from the Fisher : {marginal_sd[1]:.6f}")
print(f"1 / SNR                     : {1 / optimal_snr:.6f}")
print(f"f0-phi0 correlation         : {correlation[0, 2]:+.3f}")
print()
print(f"marginal    sigma(f0) : {marginal_sd[0]:.3e} Hz")
print(f"conditional sigma(f0) : {conditional_sd[0]:.3e} Hz")
print(f"ratio                 : {conditional_sd[0] / marginal_sd[0]:.3f}")
print(f"sqrt(1 - rho^2)       : {np.sqrt(1 - correlation[0, 2] ** 2):.3f}")'''),
        code("""# Turn the earlier one-dimensional scan into a normalised posterior and
# compare its width with both Fisher predictions.
scan_posterior = np.exp(np.array(logL) - np.max(logL))
scan_posterior /= np.trapezoid(scan_posterior, offsets)
scan_mean = np.trapezoid(scan_posterior * offsets, offsets)
scan_sd = np.sqrt(np.trapezoid(scan_posterior * (offsets - scan_mean) ** 2, offsets))

angles = np.linspace(0, 2 * np.pi, 200)
circle = np.column_stack([np.cos(angles), np.sin(angles)])
block = fisher_covariance[np.ix_([0, 2], [0, 2])]
eigenvalues, eigenvectors = np.linalg.eigh(block)

fig, (scan_ax, ellipse_ax) = plt.subplots(1, 2, figsize=(11, 3.8))
scan_ax.plot(offsets, scan_posterior, label="likelihood scan (conditional)")
gaussian = np.exp(-0.5 * (offsets / conditional_sd[0]) ** 2)
scan_ax.plot(
    offsets,
    gaussian / np.trapezoid(gaussian, offsets),
    "--",
    label="Fisher conditional",
)
wide = np.exp(-0.5 * (offsets / marginal_sd[0]) ** 2)
scan_ax.plot(
    offsets, wide / np.trapezoid(wide, offsets), ":", label="Fisher marginal"
)
scan_ax.set(
    xlabel=r"$f_0$ offset [Hz]",
    ylabel="density",
    title="Fixing a correlated parameter looks too precise",
)
scan_ax.legend(fontsize=8)

for n_sigma in (1, 2):
    ellipse = n_sigma * circle @ (eigenvectors * np.sqrt(eigenvalues)).T
    ellipse_ax.plot(ellipse[:, 0], ellipse[:, 1], color="C0")
ellipse_ax.axvline(0, color="0.7", lw=0.8)
ellipse_ax.axhline(0, color="0.7", lw=0.8)
ellipse_ax.set(
    xlabel=r"$\\Delta f_0$ [Hz]",
    ylabel=r"$\\Delta\\phi_0$ [rad]",
    title=f"Fisher ellipse, correlation {correlation[0, 2]:+.2f}",
)
plt.show()

print(f"scan sigma(f0)        : {scan_sd:.3e} Hz")
print(f"conditional sigma(f0) : {conditional_sd[0]:.3e} Hz")
print(f"marginal sigma(f0)    : {marginal_sd[0]:.3e} Hz")"""),
        md("""### The same calculation with LISA Analysis Tools

The local `lisa_analysis_workshop` calls this abstraction an
`AnalysisContainer`. It bundles a `DataResidualArray` with a compatible
`SensitivityMatrix`, then exposes inner products, SNRs, and template
likelihoods. This does not replace understanding the formula—it reduces unit,
frequency-grid, and channel bookkeeping once the formula is understood."""),
        code(
            """warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"lisatools\\..*")
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray

latw_data = DataResidualArray(template, f_arr=frequency)
latw_sensitivity = SensitivityMatrix(
    frequency,
    [A1TDISens, E1TDISens],
    stochastic_params=(t_obs,),
)
latw_analysis = AnalysisContainer(latw_data, latw_sensitivity)

best_index = int(np.argmax(logL))
offset_index = 0
latw_best_template = DataResidualArray(
    trial_templates[best_index], f_arr=frequency
)
latw_offset_template = DataResidualArray(
    trial_templates[offset_index], f_arr=frequency
)

latw_optimal_snr, latw_detected_snr = latw_analysis.template_snr(
    latw_best_template
)
latw_best_log_likelihood = latw_analysis.template_likelihood(latw_best_template)
latw_offset_log_likelihood = latw_analysis.template_likelihood(latw_offset_template)

print(f"AnalysisContainer optimal SNR: {latw_optimal_snr:.2f}")
print(f"AnalysisContainer detected SNR: {latw_detected_snr:.2f}")
print(
    "log-likelihood drop for an offset template:",
    f"{latw_offset_log_likelihood - latw_best_log_likelihood:.2f}",
)"""
        ),
        md(
            r"""## 5. The global-fit problem

The LATW challenge combines a massive-black-hole binary with groups of Galactic binaries. The GLASS demonstration writes the same idea schematically as
\[
d=h_{\rm UCB}+h_{\rm VGB}+h_{\rm MBHB}+n(\eta).
\]
No source is analysed against pristine data: each block sees a residual containing the current estimates of all other source and noise blocks. Source count may itself be unknown, motivating reversible-jump/trans-dimensional methods."""
        ),
        code(
            """frequencies=np.array([3e-3,3.00012e-3,3.00025e-3]); true_scales=np.array([1.,.72,.48])
catalogue=GBObject(f0=frequencies,fdot=np.array([1e-17,.5e-17,1.5e-17]),A=np.full(3,2e-22),ra=np.array([1.,1.4,2.]),dec=np.array([.4,-.2,.7]),psi=np.array([.3,.8,1.1]),iota=np.array([.8,1.2,.5]),phi0=np.array([.2,1.5,2.4]),t_init=0.)
all_parameters=catalogue.to_jaxgb_array(t0=0); kmins=np.asarray(simulator.get_kmin(all_parameters[:,0])); kmin=int(kmins.min()); kmax=int(kmins.max()+simulator.n)
templates=[]
for row in np.asarray(all_parameters):
    a,e,_=simulator.sum_tdi(row[None,:],kmin,kmax,tdi_generation=2,tdi_combination="AET"); templates.append(np.stack([np.asarray(a),np.asarray(e)]))
templates=np.asarray(templates); common_frequency=np.arange(kmin,kmax)/t_obs
common_psd=np.stack([get_sensitivity(common_frequency,sens_fn=A1TDISens,stochastic_params=(t_obs,)),get_sensitivity(common_frequency,sens_fn=E1TDISens,stochastic_params=(t_obs,))])
noise=np.sqrt(common_psd/(4*df))*(rng.normal(size=common_psd.shape)+1j*rng.normal(size=common_psd.shape))
data=np.sum(true_scales[:,None,None]*templates,axis=0)+noise
def global_inner(a,b): return 4*df*np.real(np.sum(np.conj(a)*b/common_psd))
fig,ax=plt.subplots(figsize=(9,3.4)); ax.plot(1e3*common_frequency,np.abs(data[0]),color="k",lw=.8,label="A-channel data")
for i,h in enumerate(templates): ax.plot(1e3*common_frequency,np.abs(true_scales[i]*h[0]),label=f"source {i+1}")
ax.set(xlabel="frequency [mHz]",ylabel="TDI A magnitude",title="Overlapping JaxGB sources plus LISA noise"); ax.legend(); plt.show()"""
        ),
        code(
            """# One sequential pass, then a blocked conditional fit.
sequential=np.zeros(3); residual=data.copy()
for i,h in enumerate(templates): sequential[i]=global_inner(h,residual)/global_inner(h,h); residual-=sequential[i]*h
blocked=np.zeros(3); history=[blocked.copy()]
for sweep in range(12):
    for i,h in enumerate(templates):
        effective=data-np.sum(blocked[:,None,None]*templates,axis=0)+blocked[i]*h
        blocked[i]=global_inner(h,effective)/global_inner(h,h)
    history.append(blocked.copy())
history=np.asarray(history)

# The simultaneous weighted least-squares solution.
whitened_templates=(np.sqrt(4*df/common_psd)[None,:,:]*templates).reshape(3,-1).T
whitened_data=(np.sqrt(4*df/common_psd)*data).ravel()
design=np.vstack([whitened_templates.real,whitened_templates.imag]); target=np.r_[whitened_data.real,whitened_data.imag]
joint=np.linalg.lstsq(design,target,rcond=None)[0]
print("true      ",np.round(true_scales,3)); print("one pass  ",np.round(sequential,3)); print("joint     ",np.round(joint,3)); print("blocked   ",np.round(blocked,3))
fig,ax=plt.subplots(figsize=(8,3.4))
for i in range(3): ax.plot(history[:,i],"o-",label=f"source {i+1}"); ax.axhline(true_scales[i],color=f"C{i}",ls="--",alpha=.5)
ax.set(xlabel="blocked sweep",ylabel="amplitude multiplier",title="Source blocks communicate through the residual"); ax.legend(); plt.show()"""
        ),
        md(
            """## 6. A miniature unknown-source-count challenge

LATW Tutorial 6 uses RJMCMC so the number of Galactic binaries is inferred. Here we enumerate all eight subsets of three candidate templates and use BIC only as a fast classroom proxy—not as a replacement for evidence or RJMCMC."""
        ),
        code(
            """model_scores=[]
for included_bits in itertools.product([0,1],repeat=3):
    included=np.flatnonzero(included_bits); model=np.zeros_like(data); n_parameters=len(included)
    if n_parameters:
        X=whitened_templates[:,included]; D=np.vstack([X.real,X.imag]); coefficients=np.linalg.lstsq(D,target,rcond=None)[0]
        model=np.sum(coefficients[:,None,None]*templates[included],axis=0)
    minus_two_logL=global_inner(data-model,data-model)
    bic=minus_two_logL+n_parameters*np.log(target.size); model_scores.append((included_bits,bic))
best=min(score for _,score in model_scores)
labels=["".join(map(str,bits)) for bits,_ in model_scores]; delta=[score-best for _,score in model_scores]
fig,ax=plt.subplots(figsize=(8,3.3)); ax.bar(labels,delta); ax.set(xlabel="included sources (1=yes)",ylabel=r"$\\Delta$BIC",title="Toy catalogue-size comparison"); plt.show()
print("Preferred subset:",labels[int(np.argmin(delta))],"(the injected subset is 111)")"""
        ),
        md("""## Extension: animate the global residual"""),
        code(
            """fig,ax=plt.subplots(figsize=(9,3.3)); line,=ax.plot([],[],lw=.8)
ax.set(xlim=(1e3*common_frequency.min(),1e3*common_frequency.max()),ylim=(0,1.1*np.max(np.abs(data[0]))),xlabel="frequency [mHz]",ylabel="A residual magnitude")
def animate_residual(i):
    residual=data-np.sum(history[i,:,None,None]*templates,axis=0); line.set_data(1e3*common_frequency,np.abs(residual[0])); ax.set_title(f"global residual after sweep {i}"); return (line,)
animation=FuncAnimation(fig,animate_residual,frames=len(history),interval=220); plt.close(fig); display(HTML(animation.to_jshtml()))"""
        ),
        md("""## Optional LISA Data Challenge input and boundary

The LDC portal requires authentication, so the live notebook uses deterministic synthetic data. An authenticated student can later upload a selected file in Colab.

```python
from google.colab import files
uploaded = files.upload()
```

This notebook uses a real sensitivity model, orbit, and TDI response, but the global exercise fits three amplitude coefficients from a fixed candidate catalogue. A research global fit must infer nonlinear parameters, source count, multiple source classes and TDI channels, instrument/foreground noise, and demonstrate convergence and coverage.

Continue with:

- local `lisa_analysis_workshop/tutorials/Tutorial1.ipynb` for sensitivity/SNR/likelihood;
- Tutorial 6 for fixed-dimensional and RJ Galactic-binary inference;
- `LATW-challenge-problem.ipynb` for an MBHB plus two trans-dimensional GB groups;
- [GLASS global analysis](https://arxiv.org/abs/2301.03673);
- [LISA Data Challenge files](https://lisa-ldc.in2p3.fr/file)."""),
    ],
)
