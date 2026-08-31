"""Build the eight Colab-first FQCP 2026 gravitational-wave PE notebooks."""

import base64
import mimetypes
import re
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

import black
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks"

# {{IMAGE:relative/path.png|alt text}} is inlined as a data URI so that local
# figures survive in Colab, which downloads only the .ipynb and not the repo.
IMAGE_TOKEN = re.compile(r"\{\{IMAGE:([^}|]+)\|([^}]*)\}\}")


def clean_source(text):
    """Remove generator indentation and surrounding blank lines."""
    return dedent(text).strip()


def inline_image(path, alt):
    """Return Markdown for a repository image embedded as a data URI."""
    target = ROOT / path.strip()
    if not target.is_file():
        raise SystemExit(f"build_course.py: missing image asset {target}")
    media_type = mimetypes.guess_type(target.name)[0] or "image/png"
    payload = base64.b64encode(target.read_bytes()).decode("ascii")
    return f"![{alt.strip()}](data:{media_type};base64,{payload})"


def md(text):
    """Create Markdown with display-math delimiters supported everywhere."""
    source = clean_source(text)
    source = source.replace(r"\[", "$$").replace(r"\]", "$$")
    source = IMAGE_TOKEN.sub(lambda m: inline_image(m.group(1), m.group(2)), source)
    return nbf.v4.new_markdown_cell(source)


# Reference figures are extracted from the executed notebooks by
# publish_assets.py and served from the force-pushed `assets` branch, so they
# cost the notebook a URL rather than ~90 KB of base64 each.
ASSET_URL = (
    "https://raw.githubusercontent.com/nz-gravity/"
    "FQCP2026_GW_data_analysis/assets/expected"
)


def code(text, figure=None):
    """Create a consistently formatted, reader-facing Python cell.

    `figure` names a reference image for this cell's output. The slug is
    recorded in cell metadata so publish_assets.py can find the output again
    after execution, and write() inserts the collapsed reference below.
    """
    source = clean_source(text)
    source = black.format_str(source, mode=black.Mode(line_length=88)).rstrip()
    cell = nbf.v4.new_code_cell(source)
    if figure:
        cell.metadata["fqcp_figure"] = figure
    return cell


def reference_block(slug):
    """Collapsed 'expected output' image, rendered by both Colab and Sphinx."""
    return nbf.v4.new_markdown_cell(
        "<details>\n"
        "<summary><i>Expected output &mdash; open this if your cell has not "
        "run yet</i></summary>\n\n"
        f'<img src="{ASSET_URL}/{slug}.png" alt="expected output: {slug}" '
        'style="max-width:100%">\n\n'
        "</details>"
    )


def write(name, title, cells, *, orphan=False):
    header = md(f"""# {title}

**FQCP 2026 · Bayesian parameter estimation for gravitational-wave sources**

> Google Colab worksheet. No prior Bayesian statistics or gravitational-wave
> experience is assumed — see the
> [glossary](https://nz-gravity.github.io/FQCP2026_GW_data_analysis/glossary.html)
> whenever a term is new. Run from top to bottom. In the JupyterBook, **Live
> route** cards identify the material for the session; **Extension** sections
> may be skipped live.
""")
    expanded = []
    for cell in [header, *cells]:
        expanded.append(cell)
        slug = cell.get("metadata", {}).get("fqcp_figure")
        if slug:
            expanded.append(reference_block(slug))
    notebook = nbf.v4.new_notebook(cells=expanded)
    # nbformat assigns a random id per cell, which would rewrite every notebook
    # on every build.  Number them by position so regeneration is a no-op in git.
    for position, cell in enumerate(notebook.cells):
        cell.id = f"cell-{position:03d}"
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": name, "provenance": []},
    }
    if orphan:
        # Build a directly addressable page without adding it to site navigation.
        notebook.metadata["orphan"] = True
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
def show_animation(animation):
    # H.264 video: ~40x smaller in the notebook than one PNG per frame
    try:
        return display(HTML(animation.to_html5_video()))
    except RuntimeError:  # ffmpeg unavailable: fall back to per-frame PNGs
        return display(HTML(animation.to_jshtml()))

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

:::{admonition} Live route — 30 minutes
:class: tip

Use the dropdowns immediately below to separate the in-room sequence from the
reference material.
:::

:::{dropdown} In the room
Sections 1–4: model, prior predictive check, likelihood/grid posterior, and
posterior predictive check. End with the PSD/Whittle bridge in Section 7.
:::
:::{dropdown} Read afterwards
Sections 5–6 explain MCMC and nested sampling; the Fisher and P–P sections are
reference material for returning to the notebook later.
:::

**One map for the whole course:**

$$\text{data} + \text{signal model} + \text{noise model}
\longrightarrow \text{likelihood} \longrightarrow \text{posterior}
\longrightarrow \text{checks} \longrightarrow \text{claim}.$$

The LVK and LISA notebooks change the data, response, and noise model—not this
logic.

We will follow the teaching sequence used in the NZ Bilby CBC workshop:

1. write a signal model;
2. choose priors;
3. write a likelihood from a noise assumption;
4. calculate a posterior on a grid;
5. inspect marginals and posterior predictions;
6. replace the grid with the algorithms real analyses use — Metropolis-Hastings
   and nested sampling — and learn how to tell whether they worked;
7. replace white noise by a gravitational-wave-style PSD-weighted likelihood.

Sections 5-7 and the extensions are written to be read on your own afterwards;
they are the reference half of this notebook and are not all covered live. A
short Fisher-matrix comparison is retained only as optional extra material.

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
        md(
            """**Predict before running:** If the same noisy data admit several plausible
lines, what information is missing from a best-fit line alone? Keep that answer
in mind when the posterior appears below."""
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
        md(
            """**Predict before running:** Which is the more serious warning sign: a prior
that is broad, or a prior predictive curve that cannot resemble the observed
data? Explain why before looking at the draw."""
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
        md(
            r"""### Code studio: write the Gaussian log likelihood

Take 5 minutes in pairs. Translate the equation above into NumPy.

- Compute the residual with `signal_model`.
- Return one scalar log likelihood.
- Keep the normalisation term: it matters when noise models are compared.
- Run the self-check. A correct function reports `check passed`.

The cell is deliberately safe to run before you fill it in."""
        ),
        code("""def student_log_likelihood(m, c):
    # YOUR CODE HERE
    return None

student_value = student_log_likelihood(true_parameters["m"], true_parameters["c"])
if student_value is None:
    print("Your turn: replace the placeholder in student_log_likelihood.")
else:
    residual = data - signal_model(time, true_parameters["m"], true_parameters["c"])
    expected = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma**2))
    np.testing.assert_allclose(student_value, expected)
    print("check passed")"""),
        md(
            r"""<details>
<summary>Show one possible solution</summary>

```python
def student_log_likelihood(m, c):
    residual = data - signal_model(time, m, c)
    return -0.5 * np.sum(
        (residual / sigma) ** 2 + np.log(2 * np.pi * sigma**2)
    )
```

</details>"""
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
axes[0].set_ylabel("intercept c"); plt.show()""", figure="basics-grid-posterior"),
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
        md(r""":::{admonition} A correct sampler cannot repair a wrong likelihood
:class: warning

Keep the same data but tell the likelihood that the noise standard deviation is
half its true value. The posterior becomes narrower because the calculation
thinks the data are more informative—not because it learned more. This is why
we check residuals and the noise/PSD model, rather than treating a sharp
posterior as success.
:::
"""),
        code("""wrong_sigma=sigma/2
wrong_logL=np.array([[-.5*np.sum(((data-signal_model(time,m,c))/wrong_sigma)**2) for c in c_grid] for m in m_grid])
wrong_posterior=np.exp(wrong_logL-wrong_logL.max()); wrong_posterior/=np.trapezoid(np.trapezoid(wrong_posterior,c_grid,axis=1),m_grid)
wrong_p_m=np.trapezoid(wrong_posterior,c_grid,axis=1)
fig,ax=plt.subplots(figsize=(7,3.2)); ax.plot(m_grid,p_m,label="correct noise model"); ax.plot(m_grid,wrong_p_m,label="assumed noise is too small"); ax.axvline(true_parameters["m"],color="k",ls="--",label="injected slope")
ax.set(xlabel="slope m",ylabel="marginal posterior density",title="Wrong noise model: overconfident, not more informed"); ax.legend(); plt.show()"""),
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
        md(
            """**Predict before running:** If we widen a prior in a direction that the data
do not constrain, should the posterior peak move, the evidence move, both, or
neither?"""
        ),
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
plt.close(fig); show_animation(learning_animation)"""),
        md(
            r"""## 4. Posterior predictive check

Draw parameter pairs from the posterior, map each through the signal model, and
**add a noise draw**. That is the posterior predictive distribution: it predicts
*data*, not the noise-free curve, so it is the only version the observed points
can actually be compared against.

A summary statistic turns the picture into a number. Take
$\chi^2=\sum_i\left[(d_i-h_i(\theta))/\sigma\right]^2$: the posterior predictive
*p*-value is the fraction of replicated datasets whose $\chi^2$ is at least as
large as the observed one. Values near 0 or 1 mean the model cannot produce data
like ours. It is a falsification test, not a score to maximise."""
        ),
        code(
            r'''def posterior_predictive(model, first, second, weights, observed, n_draw=500):
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
ax.fill_between(time, *np.quantile(replicas, [0.05, 0.95], axis=0), alpha=0.2,
                color="C0", label="90% predictive data band")
ax.fill_between(time, *np.quantile(curves, [0.05, 0.95], axis=0), alpha=0.45,
                color="C1", label="90% signal band")
ax.plot(time, data, "o", ms=3, color="k", label="data")
ax.set(xlabel="time", ylabel="observation",
       title=f"Posterior predictive, p = {p_value:.2f}")
ax.legend(fontsize=8)

rx.axhline(0, color="k", lw=1)
rx.fill_between(time, *np.quantile(replicas - curves, [0.05, 0.95], axis=0),
                alpha=0.25, color="C0", label="90% predictive residual band")
rx.plot(time, np.median(data - curves, axis=0), color="C3", label="observed residual")
rx.set(xlabel="time", ylabel="data - signal", title="No structure left over")
rx.legend(fontsize=8)
plt.show()
''', figure="basics-posterior-predictive"
        ),
        md(
            r"""### When the model is wrong

Every check so far has passed, so none of them has yet been shown to *do*
anything. Here the data are generated by an exponential,
$h(t)=A\,(e^{t/\tau}-1)$, and fitted with both the exponential model and the
straight line. The line is simply the wrong shape.

Look at the left column before reading the numbers. The wrong model does not
look absurd: the posterior is tight, the fit passes through the data, and most
points sit inside the band. **A posterior cannot tell you its model is wrong** —
it only reports the best available parameters of whatever model it was given.

The two checks that do notice are the residual panel, where the misfit appears
as a slow arc instead of scatter, and the *p*-value."""
        ),
        code(
            r'''def exponential_model(time, amplitude, tau):
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
    "linear (wrong)": (signal_model, np.linspace(-2, 6, 141), np.linspace(-15, 15, 161)),
    "exponential (true)": (exponential_model, np.linspace(0, 4, 141), np.linspace(1, 10, 161)),
}

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
evidences, p_values = {}, {}
for column, (name, (model, first_grid, second_grid)) in zip(axes.T, candidates.items()):
    first, second, weights, log_z = fit_on_grid(model, curved_data, first_grid, second_grid)
    curves, replicas, p_value = posterior_predictive(model, first, second, weights, curved_data)
    evidences[name] = log_z
    p_values[name] = p_value

    top, bottom = column
    top.fill_between(time, *np.quantile(replicas, [0.05, 0.95], axis=0), alpha=0.2, color="C0")
    top.fill_between(time, *np.quantile(curves, [0.05, 0.95], axis=0), alpha=0.5, color="C1")
    top.plot(time, curved_data, "o", ms=3, color="k")
    top.set_title(f"{name}\nlog Z = {log_z:.1f},   p = {p_value:.2f}")

    bottom.axhline(0, color="k", lw=1)
    bottom.fill_between(time, *np.quantile(replicas - curves, [0.05, 0.95], axis=0),
                        alpha=0.25, color="C0")
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
''', figure="basics-wrong-model"
        ),
        md(
            r"""The line is not merely worse, it is excluded. Its *p*-value is zero to three
decimal places: not one replicated dataset drawn from its own posterior is as
badly fitted as the real data. The evidence agrees independently, and by a wide
margin.

Two cautions that generalise to the gravitational-wave chapters:

- **The Bayes factor is not free of the priors.** These two models were given
  different prior boxes, worth about $1.9$ in log evidence. That is negligible
  against the log Bayes factor printed above, but it would not be negligible
  against a log Bayes factor of $3$.
- **A passing check is not proof.** The exponential model passes here because it
  is the model that generated the data. In a real analysis the true shape is
  never on the menu, and a *p*-value that is merely non-extreme means only that
  this particular statistic found nothing wrong. Note also that these *p*-values
  are not uniformly distributed under the true model: the data were used both to
  fit and to test, which makes them conservative.

This is the whole reason gravitational-wave analyses carry residual tests and
waveform-systematics studies alongside their posteriors."""
        ),
        md(r"""## 5. Why real PE cannot use a grid

- A grid with $n$ points per axis and $D$ parameters costs $n^D$ likelihood
  evaluations.

\[
\text{cost}=n^D,\qquad D_{\rm BBH}\approx15,\ n=20
\ \Rightarrow\ 20^{15}\approx3\times10^{19}.
\]

- At 1 ms per waveform that is about a **billion years**.
- The posterior occupies a vanishingly small fraction of that volume, so almost
  every grid point is wasted.
- Stochastic samplers spend their effort where the posterior has mass. The next
  three sections build the two that dominate gravitational-wave work:
  **MCMC** for parameters, **nested sampling** for evidence."""),
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

From the current point $\theta$, repeat:

\[
\theta'=\theta+\mathcal N(0,\Sigma_{\rm prop}),\qquad
\alpha=\min\left[1,\;
\frac{\mathcal L(\theta')\,\pi(\theta')}{\mathcal L(\theta)\,\pi(\theta)}\right],
\]

accept $\theta'$ with probability $\alpha$, otherwise **store $\theta$ again**.

- Only the *ratio* is needed, so the evidence $\mathcal Z$ cancels. This is why
  MCMC works when the normalisation is unknown.
- Repeating a rejected point is not a bug: it is how the chain accumulates
  density where the posterior is large.
- The output is a set of **correlated** draws whose histogram converges to the
  posterior. Correlated is fine; it just costs effective samples (Section 5.3).
- Uphill moves are always accepted; downhill moves are accepted sometimes. That
  is what stops the chain collapsing onto the maximum-likelihood point."""),
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

Two distinct phases to look for:

- **Burn-in** — a directed climb from the deliberately bad start in the corner,
  where the posterior is negligible. These samples describe where we started,
  not the posterior, so they are discarded.
- **Sampling** — the walker wanders up and down the degeneracy ridge. This is
  the part that is actually a draw from the posterior.

The trace panel on the right is the standard way to spot the transition."""
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
show_animation(chain_animation)"""),
        md(
            """### The proposal scale controls everything

A chain can be correct in principle and useless in practice:

- **Steps too small** — almost everything is accepted, but the walker crawls
  and never crosses the posterior. Watch the printed mean below: it is badly
  wrong, from code that has no bug.
- **Steps too large** — almost every proposal lands somewhere absurd and is
  rejected, so the chain sits still.
- **Well tuned** — acceptance around 0.2-0.3 for random-walk Metropolis.

Both failures look the same in the end: a chain that has not forgotten where it
started. `emcee`, `dynesty`, and `bilby` automate the tuning, but these failure
modes are exactly what their convergence diagnostics are looking for."""
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

Consecutive samples are correlated, so $N$ stored samples are worth fewer than
$N$ independent draws. With autocorrelation $\rho(k)$ at lag $k$,

\[
N_{\rm eff}\simeq\frac{N}{1+2\sum_{k\ge1}\rho(k)},\qquad
\text{Monte Carlo error}\propto\frac{1}{\sqrt{N_{\rm eff}}}.
\]

- $N_{\rm eff}$, **not** the raw chain length, sets the error on every
  posterior summary you quote.
- A million highly correlated samples can carry less information than a
  thousand independent ones.
- Thinning (keeping every $k$-th sample) saves storage. It does not improve
  $N_{\rm eff}$, so it never buys accuracy.
- Rule of thumb: report a number only if $N_{\rm eff}$ is in the hundreds."""),
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
print(f"sampler : m = {samples[:, 0].mean():.4f}")""", figure="basics-corner-check"),
        md(r"""## 6. Nested sampling: where the evidence comes from

MCMC gives parameters but not $\mathcal Z$. Nested sampling gives both, by
reordering the integral along **prior volume**. Let $X(\lambda)$ be the
fraction of the prior with $\mathcal L>\lambda$. Then a $D$-dimensional
integral becomes a one-dimensional one:

\[
\mathcal Z=\int\mathcal L(\theta)\,\pi(\theta)\,d\theta
=\int_0^1\mathcal L(X)\,dX
\;\simeq\;\sum_i\mathcal L_i\,\Delta X_i .
\]

The algorithm:

1. Draw $N_{\rm live}$ points from the prior.
2. Delete the **worst** one and record its likelihood.
3. Replace it by a new draw from the prior, restricted to
   $\mathcal L>\mathcal L_{\rm worst}$.
4. Repeat. Each deletion shrinks the volume by a known factor,
   $X_i\approx e^{-i/N_{\rm live}}$.

- The likelihood threshold only ever rises, so the live points contract onto
  the peak. The animation below shows exactly this.
- Step 3 is the hard part in real problems, and it is what separates
  `MultiNest` (ellipsoids) from `PolyChord` and `dynesty` (slice sampling).
  Ours evolves a copy of a surviving point with a short constrained MCMC.
- Posterior samples come out free: the deleted points weighted by
  $\mathcal L_i\Delta X_i$.
- This is why Bayesian model comparison is practical in GW astronomy at all."""),
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
show_animation(nested_animation)"""),
        md(r"""## 7. The gravitational-wave bridge: PSD and Whittle likelihood

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
        md(r"""### Optional audio analogy: hear and see what whitening does

- This is **not detector strain converted to sound**. It is an audible toy: a
  chirp buried in coloured noise.
- Whitening divides each Fourier component by the noise ASD,
  $\tilde d(f)\rightarrow\tilde d(f)/\sqrt{S_n(f)}$, which is the same
  inverse-noise weighting the Whittle likelihood applies.
- Listen first, then look at the spectrograms. The low-frequency noise wall
  carries almost all the power, which is why the raw clip sounds like rumble.
- After whitening every frequency carries comparable noise, so the chirp
  becomes the loudest and the brightest feature."""),
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
        code("""from scipy.signal import spectrogram

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
    spec_f, spec_t, power = spectrogram(series, fs=audio_rate, nperseg=256, noverlap=224)
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
plt.show()"""),
        md(r"""## Extension: the Fisher matrix

A cheap Gaussian stand-in for the posterior. Skip on a first pass; it is here
because forecast papers use it constantly and notebook 02 needs it.

- Expand $\log\mathcal L$ to second order about the peak. The posterior becomes
  a Gaussian with covariance $F^{-1}$.
- $F_{ij}=-\langle\partial_i\partial_j\log\mathcal L\rangle$, which for a
  gravitational-wave signal is $F_{ij}=(\partial_i h\mid\partial_j h)$.
- For **this** model it is not an approximation: the posterior is exactly
  Gaussian and $F=X^{\mathsf T}X/\sigma^2$ for the design matrix $X$. The
  ellipses below should sit on the grid contours.
- It breaks down at low SNR, on curved degeneracies, with multiple modes, and
  against hard prior edges. Then you need the samplers above."""),
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
        md(r""":::{admonition} Different checks answer different questions
:class: warning

| Check | What it can support | What it cannot establish alone |
| --- | --- | --- |
| trace / ESS / convergence diagnostic | whether this numerical run explored its target | whether the target model describes nature |
| posterior predictive check | whether simulated data resemble observed data in chosen summaries | coverage across repeated datasets |
| truth inside one 90% interval | a useful debugging observation | a calibrated 90% interval |
| P–P test over many simulations | end-to-end coverage under the simulated model | robustness to unmodelled systematics |

Never promote “the injected truth was inside the interval once” to a validation
claim. The P–P experiment below is the right scale of test for calibration.
:::
"""),
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
            """## Question bank and answer key

The questions above and below are deliberately collected here so that an
instructor can pause during the notebook without revealing the answer. Try to
answer them from the plots and equations first.

1. If several lines plausibly fit the data, what is missing from a best-fit line?
2. Is a broad prior or an implausible prior predictive draw the more serious
   warning sign?
3. If an unconstrained prior direction is widened, what happens to the local
   posterior peak and to the evidence?
4. Widen the prior: which marginal changes most?
5. Halve the assumed `sigma`: does the posterior become more accurate or merely more confident?
6. Why may the PSD-dependent likelihood normalisation be dropped for fixed-PSD
   PE but not when comparing noise models?
7. Change the prior width in the evidence cell. Why does the posterior near its
   peak barely move while the Bayes factor can change?
8. Start the Metropolis chain at the true parameters. Does burn-in disappear,
   and is that a safe thing to do in general?
9. Reduce `n_live` in the nested sampler. What happens to `log Z`, and why is a
   single run's evidence not enough to quote an uncertainty?
10. In the Fisher cell, shrink `sigma` by a factor of ten. Why does the ellipse
   agree with the grid posterior even better?
11. In the wrong-model section, raise `tau` towards $10$ so the exponential is
   nearly straight over the window. Which fails first, the posterior predictive
   *p*-value or the Bayes factor, and what does that tell you about how much
   data it takes to detect a modelling error?

<details>
<summary>Show the answer key</summary>

1. A best fit omits uncertainty, correlations, alternative modes, and dependence
   on modelling assumptions. The posterior supplies those missing pieces.
2. A broad prior can be reasonable; a prior predictive distribution that cannot
   produce data remotely like the observation says the model/prior combination is
   incoherent before inference begins.
3. The peak need not move, but the prior-averaged likelihood and therefore the
   evidence can fall as unsupported prior volume grows.
4. The posterior is most sensitive where the likelihood is broad or truncated by
   the prior. A well-constrained marginal near the likelihood peak changes less.
5. Halving `sigma` makes the calculation more confident. It makes it more
   accurate only if the smaller noise scale is actually the data-generating one.
6. The PSD normalisation is constant in parameters for fixed-PSD PE, but it
   changes between competing noise models and must then be retained.
7. Widening an unconstrained prior need not move a local posterior peak, but it
   lowers the prior-averaged likelihood and can lower the evidence.
8. Starting near the truth can shorten a visible transient in this toy problem,
   but it is unsafe in general: the truth is unknown and a favourable start can
   hide poor mixing or missed modes.
9. Fewer live points make nested-sampling evidence noisier and less reliable;
   one run cannot by itself establish its numerical uncertainty.
10. At higher signal-to-noise ratio the posterior is more nearly local and
   Gaussian, so the Fisher ellipse better approximates the exact grid result.
11. Both weaken as the two models become harder to tell apart, because the
   curvature the data must resolve shrinks. A modelling error is only detectable
   when it is large compared with the noise, so the same wrong waveform can be
   harmless at low SNR and fatal at high SNR. This is exactly why waveform
   systematics matter more as detectors improve.
</details>

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
        md(r""":::{admonition} Live route — 45 minutes
:class: tip

Use the dropdowns immediately below to separate the in-room sequence from the
reference material.
:::

:::{dropdown} In the room
Sections 1–5: CBC parameters, detector response, matched filtering, the manual
likelihood, and network localisation. Run the distance–inclination posterior.
:::
:::{dropdown} Read afterwards
Section 6 is a compact population-inference bridge; Section 7 is a genuine
Bilby/dynesty run and can take a few minutes in Colab.
:::

:::{admonition} Do not collapse these three questions
:class: important

| Task | Question | Typical output |
| --- | --- | --- |
| detection/search | Is there a candidate inconsistent with noise? | trigger, ranking statistic, false-alarm control |
| parameter estimation | Which source parameters remain plausible? | posterior, credible intervals, posterior predictions |
| population inference | What generated many detected events? | hyperposterior, selection-aware population model |

Matched filtering in Section 3 is the **search** bridge. The likelihood in
Section 4 begins **parameter estimation**. Section 6 asks the population
question and must account for what the detectors were able to find.
:::
"""),
        code(
            """import os,sys,subprocess,importlib.util
IN_COLAB="COLAB_RELEASE_TAG" in os.environ
missing=[p for p in ("ripplegw","bilby","gwpy") if importlib.util.find_spec(p) is None]
if missing:
    if IN_COLAB: subprocess.check_call([sys.executable,"-m","pip","install","-q","rippleGW==0.2.1","bilby==2.8.0","gwpy>=3.0,<4"])
    else: raise ImportError("Install rippleGW==0.2.1, bilby==2.8.0, and gwpy>=3.0,<4, or run in Colab.")"""
        ),
        code("""import logging
import numpy as np
import matplotlib.pyplot as plt
import bilby
from gwpy.timeseries import TimeSeries
from IPython.display import HTML,display
from matplotlib.animation import FuncAnimation
from jax import config
config.update("jax_enable_x64",True)
import jax.numpy as jnp
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
logging.getLogger("bilby").setLevel(logging.ERROR)
plt.style.use("seaborn-v0_8-whitegrid")
def show_animation(animation):
    # H.264 video: ~40x smaller in the notebook than one PNG per frame
    try:
        return display(HTML(animation.to_html5_video()))
    except RuntimeError:  # ffmpeg unavailable: fall back to per-frame PNGs
        return display(HTML(animation.to_jshtml()))
rng=np.random.default_rng(20260817)

from matplotlib.ticker import NullFormatter, ScalarFormatter


def tidy_log_frequency(axis, ticks=(20, 50, 100, 200, 500)):
    \"\"\"Readable Hz labels: matplotlib's log minor ticks overlap on wide bands.\"\"\"
    axis.set_xticks(list(ticks))
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.xaxis.set_minor_formatter(NullFormatter())"""),
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
axes[1].semilogx(frequency[mask],np.unwrap(np.angle(injection_polarizations["plus"][mask])))
axes[1].set(xlabel="frequency [Hz]",ylabel="phase [rad]",title="Hundreds of radians accumulate in band")
for ax in axes: tidy_log_frequency(ax)
plt.show()""", figure="lvk-network-response"
        ),
        md(
            r"""For a non-precessing circular binary, approximately
$h_+\propto(1+\cos^2\iota)/(2D_L)$ and $h_\times\propto\cos\iota/D_L$.
Inclination is the binary's orientation to us; polarisation angle rotates the plus/cross basis on the sky."""
        ),
        md(r"""### Animation: why chirp mass is measured so precisely

The signal accumulates hundreds of radians of phase in band, so a wrong chirp
mass shows up as **dephasing**, not as a wrong amplitude.

A template is free to slide in time and shift its overall phase, and matched
filtering maximises over both. So the fair comparison removes a linear-in-$f$
term first:

\[
\Delta\psi(f)=\psi_{\mathcal M}(f)-\psi_{\mathcal M_{\rm true}}(f)
-\underbrace{(a f+b)}_{\text{absorbed by }t_c,\ \phi_c}.
\]

What is left cannot be absorbed and is what destroys the match.

- Left: the amplitude barely moves. You could not measure $\mathcal M$ this way.
- Right: the residual dephasing. Once $|\Delta\psi|$ exceeds about 1 radian
  (grey band) the template and signal drift out of step and the recovered SNR
  falls, as Section 3 will show directly."""),
        code(
            """mass_offsets = np.linspace(-2.0, 2.0, 21)
mc_true = float(theta_true[0])
f_band = frequency[mask]
reference = injection_polarizations["plus"][mask]
reference_phase = np.unwrap(np.angle(reference))
# Weight the tc/phi_c fit by signal power so it reflects where the SNR is.
weight = np.abs(reference) ** 2
basis = np.vstack([f_band, np.ones_like(f_band)]).T
weighted_basis = basis * np.sqrt(weight)[:, None]

fig, (amp_ax, phase_ax) = plt.subplots(1, 2, figsize=(11, 3.5), dpi=80)
amp_ax.loglog(f_band, np.abs(reference), color="0.7", lw=3, label="injection")
(amp_line,) = amp_ax.loglog([], [], color="C0", label="trial template")
amp_ax.set(
    xlim=(20, 512),
    ylim=(1e-25, 3e-22),
    xlabel="frequency [Hz]",
    ylabel=r"$|h_+|$",
    title="amplitude: almost no information",
)
amp_ax.legend(loc="lower left", fontsize=8)

phase_ax.axhspan(-1, 1, color="0.8", alpha=0.7)
phase_ax.axhline(0, color="k", lw=0.8)
(phase_line,) = phase_ax.semilogx([], [], color="C3")
phase_ax.set(
    xlim=(20, 512),
    ylim=(-10, 10),
    xlabel="frequency [Hz]",
    ylabel=r"$\\Delta\\psi$ [rad]",
    title="residual dephasing: all the information",
)
for ax in (amp_ax, phase_ax):
    tidy_log_frequency(ax)
fig.subplots_adjust(top=0.80, wspace=0.28)


def animate_mass(i):
    trial = polarizations(theta_true.at[0].set(mc_true + mass_offsets[i]))["plus"][mask]
    difference = np.unwrap(np.angle(trial)) - reference_phase
    coefficients = np.linalg.lstsq(
        weighted_basis, difference * np.sqrt(weight), rcond=None
    )[0]
    residual = difference - basis @ coefficients
    amp_line.set_data(f_band, np.abs(trial))
    phase_line.set_data(f_band, residual)
    fig.suptitle(
        f"chirp mass error {mass_offsets[i]:+.2f} solar masses "
        f"({100 * mass_offsets[i] / mc_true:+.1f}%), "
        f"peak dephasing {np.abs(residual).max():.1f} rad"
    )
    return amp_line, phase_line


mass_animation = FuncAnimation(
    fig, animate_mass, frames=len(mass_offsets), interval=160
)
plt.close(fig)
show_animation(mass_animation)"""
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
plt.close(fig); show_animation(inspiral_animation)"""),
        md(r"""## 2. From source to a detector network

Each detector sees one projected combination of the two polarisations, delayed
by its own light-travel time:

\[
\tilde h_I=\left[F^I_+(\alpha,\delta,\psi,t)\,h_+
+F^I_\times(\alpha,\delta,\psi,t)\,h_\times\right]e^{-2\pi if\Delta t_I}.
\]

If the detector noises are independent given their PSDs, the network likelihood
is a product, so log-likelihoods simply add:

\[
\log\mathcal L_{\rm net}=\sum_I\log\mathcal L_I
=-\frac12\sum_I(d_I-h_I\mid d_I-h_I)_I+C.
\]

- $F_+^I,F_\times^I$ depend on sky position, polarisation, detector
  orientation, and sidereal time. Bilby stores the geometry and applies this.
- $\Delta t_I$ is the arrival-time delay, and differences between detectors are
  what localise the source (Section 5).
- **Source parameters are shared**; only the response and the noise weighting
  are detector-specific. That is the whole reason a network beats one detector.
- Adding a detector adds its $(d\mid h)$ terms, so SNRs add in quadrature."""),
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
for ax in axes:
    tidy_log_frequency(ax)
    ax.legend()
plt.show()"""),
        md(r"""### All-sky network response: predict, then inspect

**Predict before running:** Does adding a detector make every sky direction
equally loud, or does it mainly fill particular blind spots? Which information
needed for localisation is absent from a sensitivity map?

Each detector's polarisation-averaged sensitivity to a direction is

\[
R_I(\alpha,\delta)=\sqrt{F_+^{I\,2}+F_\times^{I\,2}},\qquad
\text{network proxy}=\sqrt{\sum_I\left(\frac{R_I}{\mathrm{ASD}_I}\right)^2}.
\]

- $R_I$ is **independent of the polarisation angle** $\psi$: rotating $\psi$
  mixes $F_+$ and $F_\times$ but preserves this combination.
- A single interferometer has a quadrupolar pattern with four blind spots.
  Watch how the three individual maps put their blind spots in *different*
  places, so the network map is far more uniform.
- The proxy is noise-weighted with each ASD at 100 Hz. It is not a real SNR:
  that also needs the waveform, distance, inclination, and full PSD.
- A sensitivity map says how *loud* a source is, not *where* it is. Localisation
  comes from arrival-time and phase differences, which Section 5 covers."""),
        code("""sky_ra = np.linspace(-np.pi, np.pi, 73)
sky_dec = np.linspace(-np.pi / 2, np.pi / 2, 37)
sky_ra_grid, sky_dec_grid = np.meshgrid(sky_ra, sky_dec)
reference_frequency = 100.0
reference_index = np.argmin(np.abs(frequency - reference_frequency))
asd_reference = np.array(
    [
        ifo.power_spectral_density.get_amplitude_spectral_density_array(frequency)[
            reference_index
        ]
        for ifo in ifos
    ]
)


def response_and_snr_proxy(ra, dec):
    responses = []
    for ifo in ifos:
        f_plus = ifo.antenna_response(ra, dec, gps_time, source_parameters["psi"], "plus")
        f_cross = ifo.antenna_response(ra, dec, gps_time, source_parameters["psi"], "cross")
        responses.append(np.hypot(f_plus, f_cross))
    responses = np.asarray(responses)
    return responses, np.sqrt(np.sum((responses / asd_reference) ** 2, axis=0))


# Per-detector response maps and the noise-weighted network proxy.
detector_maps = np.array(
    [
        [[response_and_snr_proxy(ra, dec)[0][k] for ra in sky_ra] for dec in sky_dec]
        for k in range(len(ifos))
    ]
)
snr_proxy_map = np.array(
    [[response_and_snr_proxy(ra, dec)[1] for ra in sky_ra] for dec in sky_dec]
)
snr_proxy_scale = snr_proxy_map.max()
detector_names = [ifo.name for ifo in ifos]

detector_scales = detector_maps.max(axis=(1, 2))
panels = [
    (name, detector_maps[k] / detector_scales[k])
    for k, name in enumerate(detector_names)
]
panels.append(("network", snr_proxy_map / snr_proxy_scale))

# Four reasonably sized maps reveal the blind spots more clearly than one
# compressed row. The bar chart answers the local question at the marker.
fig = plt.figure(figsize=(12, 6.8), dpi=78)
grid_spec = fig.add_gridspec(
    2, 3, width_ratios=[1, 1, 0.72], hspace=0.24, wspace=0.16
)
markers = []
map_axes = []
for panel_index, (name, field) in enumerate(panels):
    row, column = divmod(panel_index, 2)
    sky_ax = fig.add_subplot(grid_spec[row, column], projection="mollweide")
    map_axes.append(sky_ax)
    image = sky_ax.pcolormesh(
        sky_ra_grid, sky_dec_grid, field, shading="auto", cmap="viridis", vmin=0, vmax=1
    )
    (marker,) = sky_ax.plot([], [], "o", color="C3", mec="white", ms=7)
    markers.append(marker)
    sky_ax.grid(True, lw=0.4, alpha=0.5)
    sky_ax.set_xticklabels([])
    sky_ax.set_yticklabels([])
    sky_ax.set_title(
        f"{name} blind spots" if name != "network" else "network (noise-weighted)",
        fontsize=10,
    )
fig.colorbar(
    image,
    ax=map_axes,
    location="bottom",
    pad=0.08,
    shrink=0.72,
    label="normalised response",
)

sky_ra_frames = np.linspace(-np.pi, np.pi, 24, endpoint=False)
response_ax = fig.add_subplot(grid_spec[:, 2])
bars = response_ax.bar(
    [*detector_names, "network"],
    np.zeros(4),
    color=["C0", "C1", "C2", "0.25"],
)
response_ax.set(
    ylim=(0, 1.05),
    ylabel="normalised response at marker",
    title="what this sky position gives",
)
response_ax.tick_params(axis="x", rotation=30)


def animate_sky_response(frame):
    source_ra = sky_ra_frames[frame]
    for marker in markers:
        marker.set_data([source_ra], [source_parameters["dec"]])
    responses, network_response = response_and_snr_proxy(
        source_ra, source_parameters["dec"]
    )
    values = np.r_[responses / detector_scales, network_response / snr_proxy_scale]
    for bar, value in zip(bars, values):
        bar.set_height(value)
    fig.suptitle(
        f"source at right ascension {source_ra:+.2f} rad; "
        f"network proxy {values[-1]:.2f}"
    )
    return (*markers, *bars)


response_animation = FuncAnimation(
    fig, animate_sky_response, frames=len(sky_ra_frames), interval=140
)
plt.close(fig)
show_animation(response_animation)"""),
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
            r"""### Animation: sliding the template through whitened data

- Whitening divides each Fourier bin by the noise ASD, so every frequency
  carries comparable noise. This is the weighting the likelihood applies.
- Left: the whitened template slides across the whitened H1 data.
  Right: the SNR that the overlap produces at each shift.
- The signal is invisible by eye, yet the filter finds it: the template adds
  the signal **coherently** over hundreds of cycles while noise adds
  incoherently. That is the $\sqrt{N_{\rm cycles}}$ gain."""
        ),
        md(
            """**Predict before running:** Why can the filter find a signal that is
invisible by eye in the whitened data, and why would an incorrect phase
evolution stop that coherent accumulation?"""
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
show_animation(filter_animation)"""),
        md(
            """### A template only works if it is close enough

- A search cannot use the true waveform: the true parameters are what we are
  looking for. It uses a **bank** of templates and hopes one is close enough.
- Below, the same data are filtered with deliberately wrong chirp masses.
- How fast the recovered SNR falls sets how densely the bank must be packed.
  Banks are built to lose no more than a few percent of SNR anywhere.
- Compare with the dephasing animation in Section 1: the SNR loss here is that
  dephasing, integrated over the band."""
        ),
        md(
            """### Code studio: build a tiny template bank

Write a function that loops over chirp-mass offsets, builds each trial waveform,
runs `matched_filter`, and stores the largest recovered SNR. Use only the
objects already defined above. The peak should lie close to zero offset."""
        ),
        code("""def student_template_bank_scan(offsets):
    # YOUR CODE HERE
    return None

student_bank = student_template_bank_scan(np.linspace(-2, 2, 9))
if student_bank is None:
    print("Your turn: return one peak SNR for every trial chirp-mass offset.")
else:
    student_bank = np.asarray(student_bank)
    assert student_bank.shape == (9,)
    best_offset = np.linspace(-2, 2, 9)[np.argmax(student_bank)]
    assert abs(best_offset) <= 0.5
    print(f"check passed; best template offset = {best_offset:+.1f} solar masses")"""),
        md(
            r"""<details>
<summary>Show one possible solution</summary>

```python
def student_template_bank_scan(offsets):
    peaks = []
    for offset in offsets:
        trial = polarizations(theta_true.at[0].set(float(theta_true[0]) + offset))
        snr_series, _ = matched_filter(h1, trial)
        peaks.append(snr_series.max())
    return np.asarray(peaks)
```

</details>"""
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
        md(r"""## 4. Inject and infer manually

We free only the detector-frame chirp mass $\mathcal M$:

\[
p(\mathcal M\mid d)\propto \pi(\mathcal M)
\exp\!\left[-\frac12\sum_I
(d_I-h_I(\mathcal M)\mid d_I-h_I(\mathcal M))_I\right].
\]

- The waveform changes once, then is projected into H1, L1, and Virgo.
- Independent detector log likelihoods add to form the network likelihood.
- We use zero-noise data so the width is deterministic; the PSD still sets the
  expected uncertainty.
- The grid is deliberately zoomed to $\pm0.1\,M_\odot$. On the old wide scale
  both posteriors were visually indistinguishable spikes.

Replace `set_strain_data_from_zero_noise` with Bilby's PSD-noise method to study
scatter across noise realisations."""),
        code(
            """for ifo in ifos: ifo.inject_signal_from_waveform_polarizations(source_parameters,injection_polarizations)
print("Network optimal SNR:",round(np.sqrt(sum(ifo.meta_data["optimal_SNR"]**2 for ifo in ifos)),2))

def detector_log_likelihood(ifo,model_polarizations):
    model=ifo.get_detector_response(model_polarizations,source_parameters,frequencies=frequency)
    residual=ifo.frequency_domain_strain-model
    psd=ifo.power_spectral_density_array
    return -2*df*np.sum(np.abs(residual[mask])**2/psd[mask])

# The chirp mass is measured to ~0.01 solar masses, so the grid must be narrow.
# A +/-2 solar mass window would be about 120 sigma wide and show only a spike.
mass_grid=np.linspace(float(theta_true[0])-.1,float(theta_true[0])+.1,141)
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

def summarise(density_values):
    mean=np.trapezoid(density_values*mass_grid,mass_grid)
    return np.sqrt(np.trapezoid(density_values*(mass_grid-mean)**2,mass_grid))

sd_h1=summarise(posterior_h1); sd_network=summarise(posterior_network)
snr_h1=ifos[0].meta_data["optimal_SNR"]
snr_network=np.sqrt(sum(ifo.meta_data["optimal_SNR"]**2 for ifo in ifos))

fig,ax=plt.subplots(figsize=(8,3.4)); ax.plot(mass_grid,posterior_h1,label=f"H1 only (SNR {snr_h1:.1f})")
ax.plot(mass_grid,posterior_network,label=f"H1+L1+V1 (SNR {snr_network:.1f})"); ax.axvline(float(theta_true[0]),color="k",ls="--",label="injection")
ax.set(xlabel="detector-frame chirp mass [solar masses]",ylabel="posterior density",title="A coherent network gives more information"); ax.legend(); plt.show()

print(f"sigma, H1 only : {sd_h1:.4f} solar masses")
print(f"sigma, network : {sd_network:.4f} solar masses")
print(f"width ratio    : {sd_h1/sd_network:.2f}")
print(f"SNR ratio      : {snr_network/snr_h1:.2f}  <- posterior width scales as 1/SNR")"""
        ),
        md("""### Put the same likelihood behind Bilby's interface

- A Bilby likelihood is just a class with a `log_likelihood` method and a
  declared parameter set. Nothing is hidden.
- Here Bilby wraps the **exact** network calculation written above.
- The assertion checks the library interface against the manual values, so the
  transition from hand-rolled to production code is verified, not assumed."""),
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

**Predict before running:** If we double the distance, which change in
inclination could approximately restore the observed amplitude? What feature
should that create in the joint posterior?

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
)""", figure="lvk-distance-inclination"),
        md(
            r"""## 5. Why a network localises the sky

For detectors at positions $\mathbf x_I$ and $\mathbf x_J$, a sky direction
$\hat{\mathbf n}$ predicts

\[
\Delta t_{IJ}(\hat{\mathbf n})
=\frac{\hat{\mathbf n}\cdot(\mathbf x_I-\mathbf x_J)}{c}.
\]

- **One detector:** no time difference, so timing alone allows the whole sky.
- **Two detectors:** one measured delay selects a ring of constant
  $\Delta t_{IJ}$.
- **Three detectors:** two independent delays intersect into much smaller
  regions.

Real Bilby localisation also uses coherent phase, antenna amplitudes,
polarisation, distance-inclination correlations, waveform uncertainty, and sky
priors."""
        ),
        md(
            """**Predict before running:** Why does one arrival-time difference make a
ring, rather than a point? When Virgo is added, which degeneracy remains because
this particular calculation still uses timing alone?"""
        ),
        code(
            """ra=np.linspace(-np.pi,np.pi,91); dec=np.linspace(-np.pi/2,np.pi/2,46); RA,DEC=np.meshgrid(ra,dec)
delays={ifo.name:np.array([[ifo.time_delay_from_geocenter(r,d,gps_time) for r in ra] for d in dec]) for ifo in ifos}
observed={ifo.name:ifo.time_delay_from_geocenter(source_parameters["ra"],source_parameters["dec"],gps_time) for ifo in ifos}; sigma_t=3e-4
def timing_likelihood(names):
    # With a single detector there is no arrival-time difference to form, so
    # the timing likelihood is flat: every direction is equally allowed.
    ref=names[0]; value=np.zeros_like(RA)
    for name in names[1:]: value-=.5*((delays[name]-delays[ref]-(observed[name]-observed[ref]))/sigma_t)**2
    return value
panels=[(["H1"],"one detector:\\nno timing information"),
        (["H1","L1"],"two detectors:\\na ring of constant delay"),
        (["H1","L1","V1"],"three detectors:\\nring intersections")]
fig,axes=plt.subplots(1,3,figsize=(15,3.8),subplot_kw={"projection":"mollweide"})
for ax,(names,title) in zip(axes,panels):
    ll=timing_likelihood(names); sky=np.exp(ll-ll.max()); ax.contourf(RA,DEC,sky,levels=np.linspace(.05,1,15),cmap="magma")
    ax.plot(source_parameters["ra"],source_parameters["dec"],"c*",ms=10); ax.set_title(title,fontsize=10); ax.grid(True,lw=.4,alpha=.5)
    ax.set_xticklabels([]); ax.set_yticklabels([])
plt.show()
print("Sky area allowed by timing alone shrinks with each added detector.")
print("One detector constrains direction only through its antenna pattern,")
print("which is why a single-detector alert has a nearly all-sky map.")""", figure="lvk-sky-localisation"
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
            """This compact example treats masses as exactly measured. Real population inference reweights uncertain event posteriors, estimates selection with injection campaigns, infers several hyperparameters and often the rate, and checks sensitivity to event-level priors and waveform systematics."""
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
        md(r""":::{admonition} Live route — 40 minutes
:class: tip

Use the dropdowns immediately below to separate the in-room sequence from the
reference material.
:::

:::{dropdown} In the room
Sections 1–4: moving response, sensitivity/foreground, the complications lab,
and a manual likelihood. Then follow the global-fit wheel in Section 5.
:::
:::{dropdown} Read afterwards
The Fisher extension, the package-level route in Section 4b, the realistic
miniature fit, and unknown-source-count challenge are designed for follow-up.
:::
"""),
        code(
            """import os,sys,subprocess,importlib.util
IN_COLAB="COLAB_RELEASE_TAG" in os.environ
needed=("lisatools","gpubackendtools","jaxgb","eryn","wdm_transform")
if any(importlib.util.find_spec(package) is None for package in needed):
    if IN_COLAB:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","lisaanalysistools==1.2.5","gpubackendtools==0.1.1","jaxgb==0.2.1","astropy==7.2.0","eryn==1.2.6","wdm-transform==0.5.0"])
    else: raise ImportError("Install the pinned LISA requirements, or run in Colab.")"""
        ),
        code(
            '''import itertools
import time
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
from lisaorbits import EqualArmlengthOrbits,KeplerianOrbits,LINKS
from lisaorbits.utils import emitter,receiver
from lisaconstants import c as C_SI
from jaxgb.jaxgb import JaxGB
from jaxgb.params import GBObject
from wdm_transform import TimeSeries as WDMTimeSeries, WDM, wdm_inner_product, wdm_noise_variance
rng=np.random.default_rng(20260817); plt.style.use("seaborn-v0_8-whitegrid")
def show_animation(animation):
    # H.264 video: ~40x smaller in the notebook than one PNG per frame
    try:
        return display(HTML(animation.to_html5_video()))
    except RuntimeError:  # ffmpeg unavailable: fall back to per-frame PNGs
        return display(HTML(animation.to_jshtml()))'''
        ),
        md(
            r"""## 1. LISA's band and source zoo

Ground-based detectors observe roughly tens of Hz to kHz. LISA targets approximately $10^{-4}$–$10^{-1}$ Hz, containing Galactic compact binaries, massive-black-hole binaries, EMRIs, stellar-origin binaries, stochastic backgrounds, and instrument noise. Long observations make many signals overlap.

Unlike a static right-angle detector, LISA is a heliocentric triangle that cartwheels as it orbits. Six delayed one-way laser links are combined into time-delay interferometry (TDI) variables. Orbital modulation helps localisation, while finite arms create a frequency-dependent response."""
        ),
        md(r""":::{admonition} What is the LISA data object?
:class: important

$$\text{inter-spacecraft phase measurements}
\longrightarrow \text{delayed TDI combinations}
\longrightarrow (A,E,T)\ \text{channels}
\longrightarrow \text{response + PSD}
\longrightarrow \text{likelihood}.$$

TDI is not a cosmetic re-labelling of a strain time series: delayed link
measurements cancel laser frequency noise and define the channels whose
response and noise enter inference. In the simple likelihood below we use A and
E as independent channels; this is an analysis approximation to state and
check, not a property of every possible data product.
:::
"""),
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
        md(
            """### What the mixed data stream looks like

These LISA Data Challenge views make the overlap concrete. The same year of
Sangria data contains persistent Galactic structure, instrument noise, and
shorter massive-black-hole-binary signals.

| Time-domain mixture | Time-frequency view |
| --- | --- |
| ![Sangria time-domain data showing instrument noise, the full Galaxy, verification binaries, and massive black-hole binaries](https://lisa-ldc.in2p3.fr/static/data/img/Sangria_TD.png) | ![Sangria time-frequency periodogram with massive-black-hole-binary signals annotated](https://lisa-ldc.in2p3.fr/static/data/img/PeriodogramAnn.png) |

*Official LDC2A Sangria illustrations from the
[LISA Data Challenge](https://lisa-ldc.in2p3.fr/). The key lesson is visual:
there is no pristine data segment belonging to only one source class.*"""
        ),
        md(
            """### 1a. Choose the orbit model

The orbit object supplies spacecraft positions, directed link vectors, and
retarded light-travel times. The default below keeps the approximately
equal-arm configuration. To regenerate the link and TDI data with flexing,
unequal arms, comment the first assignment and uncomment the second, then rerun
from this cell."""
        ),
        code(
            """year=YRSID_SI; AU=149597870700.

# Default executed configuration: approximately equal arms.
orbits=EqualArmlengthOrbits()

# BREATHING-ORBIT RE-RUN: comment the line above, uncomment this one, and
# rerun from here through the XYZ/AET cells below.
# orbits=KeplerianOrbits()

print(f"active orbit model: {type(orbits).__name__}")
times=np.linspace(0,year,240)
positions=np.asarray(orbits.compute_position(times,[1,2,3])); fig,ax=plt.subplots(figsize=(5.4,5.4))
for i,label in enumerate(["spacecraft 1","spacecraft 2","spacecraft 3"]): ax.plot(positions[:,i,0]/AU,positions[:,i,1]/AU,label=label)
ax.plot(0,0,"o",color="gold",mec="k",label="Sun"); ax.set(xlabel="heliocentric x [AU]",ylabel="heliocentric y [AU]",title="An explicit LISA orbit model",aspect="equal"); ax.legend(); plt.show()"""
        ),
        md(
            """### Animation: orbital motion becomes response modulation

The binary in this animation is fixed on the sky. LISA's changing orientation
changes the antenna projection, while its motion around the Sun changes the
arrival-time phase. Those two measured modulations—not the orbit picture by
itself—are what help encode sky position.

The right-hand curves use a **low-frequency Michelson response proxy** to make
the causal link visible. The exact delayed-link and TDI responses are built in
the cells immediately below."""
        ),
        md(
            """**Predict before running:** Hold a binary fixed on the sky. Which observed
features can change during a year even if the binary itself is nearly
monochromatic, and why can those changes help localisation?"""
        ),
        code(
            """# Fixed source and polarisation tensors used only for this response-intuition
# panel. The full finite-arm, delayed-link calculation follows below.
animation_ra,animation_dec=1.0,.4
animation_source=np.array([np.cos(animation_dec)*np.cos(animation_ra),
                           np.cos(animation_dec)*np.sin(animation_ra),
                           np.sin(animation_dec)])
animation_k=-animation_source
animation_p=np.cross(animation_k,np.array([0.,0.,1.])); animation_p/=np.linalg.norm(animation_p)
animation_q=np.cross(animation_k,animation_p)
animation_plus=np.outer(animation_p,animation_p)-np.outer(animation_q,animation_q)
animation_cross=np.outer(animation_p,animation_q)+np.outer(animation_q,animation_p)

def unit_rows(values): return values/np.linalg.norm(values,axis=1)[:,None]
animation_arms=[unit_rows(positions[:,j]-positions[:,i]) for i,j in [(0,1),(1,2),(2,0)]]
n12,n23,n31=animation_arms
n13=-n31
michelson_tensor=.5*(np.einsum("ni,nj->nij",n12,n12)-np.einsum("ni,nj->nij",n13,n13))
f_plus=np.einsum("nij,ij->n",michelson_tensor,animation_plus)
f_cross=np.einsum("nij,ij->n",michelson_tensor,animation_cross)
response_envelope=np.hypot(f_plus,f_cross)
animation_barycentre=positions.mean(axis=1)
animation_frequency=3e-3
doppler_phase_cycles=animation_frequency*(animation_barycentre@animation_source)/C_SI
arm_projections=np.asarray([
    np.abs(np.einsum("ni,ij,nj->n",arm,animation_plus,arm))
    for arm in animation_arms
])

# The real triangle is tiny on an AU-scale plot. Enlarge it around its true
# barycentre so the cartwheel can be seen, and label that display choice.
DISPLAY_TRIANGLE_SCALE=8
display_positions=(animation_barycentre[:,None,:]
                   +DISPLAY_TRIANGLE_SCALE*(positions-animation_barycentre[:,None,:]))
orbit_frames=np.arange(0,len(times),5)
fig=plt.figure(figsize=(11,5.2))
grid=fig.add_gridspec(2,2,width_ratios=(1.05,1),hspace=.38)
orbit_ax=fig.add_subplot(grid[:,0]); response_ax=fig.add_subplot(grid[0,1]); phase_ax=fig.add_subplot(grid[1,1])
orbit_ax.plot(animation_barycentre[:,0]/AU,animation_barycentre[:,1]/AU,color=".75",lw=1)
orbit_ax.plot(0,0,"o",color="gold",mec="k",ms=9)
source_arrow=animation_source[:2]/np.linalg.norm(animation_source[:2])
orbit_ax.arrow(0,0,.43*source_arrow[0],.43*source_arrow[1],width=.008,
               color="C3",length_includes_head=True)
orbit_ax.text(.47*source_arrow[0],.47*source_arrow[1],"fixed source",color="C3",ha="center")
spacecraft_points,=orbit_ax.plot([],[],"o",color="k",ms=4)
arm_lines=[orbit_ax.plot([],[],lw=3)[0] for _ in range(3)]
orbit_ax.text(.03,.03,f"triangle size x{DISPLAY_TRIANGLE_SCALE} for display",
              transform=orbit_ax.transAxes,fontsize=8,color=".35")
orbit_ax.set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),aspect="equal",xlabel="x [AU]",ylabel="y [AU]")

mission_days=times/86400
response_ax.plot(mission_days,response_envelope,color="C0")
response_marker,=response_ax.plot([],[],"o",color="C0",ms=7)
response_ax.set(xlim=(0,mission_days[-1]),ylim=(0,1.08*response_envelope.max()),
                ylabel="antenna amplitude",title="same binary, changing projection")
phase_ax.plot(mission_days,doppler_phase_cycles,color="C3")
phase_marker,=phase_ax.plot([],[],"o",color="C3",ms=7)
phase_ax.axhline(0,color=".8",lw=.8)
phase_ax.set(xlim=(0,mission_days[-1]),
             ylim=(1.08*doppler_phase_cycles.min(),1.08*doppler_phase_cycles.max()),
             xlabel="mission time [days]",ylabel="Doppler phase [cycles]")

def animate_response(frame):
    i=orbit_frames[frame]
    current=display_positions[i,:,:2]/AU
    spacecraft_points.set_data(current[:,0],current[:,1])
    for arm_index,(first,second) in enumerate([(0,1),(1,2),(2,0)]):
        arm_lines[arm_index].set_data(current[[first,second],0],current[[first,second],1])
        arm_lines[arm_index].set_color(plt.cm.viridis(arm_projections[arm_index,i]))
    response_marker.set_data([mission_days[i]],[response_envelope[i]])
    phase_marker.set_data([mission_days[i]],[doppler_phase_cycles[i]])
    orbit_ax.set_title(f"LISA at day {mission_days[i]:.0f}; arm colour = GW projection")
    return (*arm_lines,spacecraft_points,response_marker,phase_marker)

orbit_response_animation=FuncAnimation(fig,animate_response,frames=len(orbit_frames),interval=110)
plt.close(fig); show_animation(orbit_response_animation)"""
        ),
        md(
            r"""### 1b. Orbits become six directed delays

For a measurement received at time $t$ on spacecraft $j$, a photon was emitted
from spacecraft $i$ at the retarded time $t-L_{ij}(t)$, where
$L_{ij}$ is the **light-travel time in seconds**:

$$
L_{ij}(t)=t_{\rm receive}-t_{\rm emit},\qquad
\hat{\mathbf n}_{ij}(t)=
\frac{\mathbf x_j(t)-\mathbf x_i(t-L_{ij})}
{cL_{ij}(t)}.
$$

The two directions along one geometric arm are distinct links. Even an
approximately equal-arm moving constellation has small directional differences
from the constellation motion. `KeplerianOrbits` adds visible arm flexing."""
        ),
        code(
            """link_codes=np.asarray(LINKS)
link_labels=[f"{int(emitter(code))} to {int(receiver(code))}" for code in link_codes]
delay_times=np.linspace(0,year,366)
selected_light_times=np.asarray(orbits.compute_ltt(delay_times,link_codes))

# This cheap reference comparison shows breathing without changing the active
# data-generation configuration selected above.
equal_reference=EqualArmlengthOrbits()
breathing_reference=KeplerianOrbits()
equal_delays=np.asarray(equal_reference.compute_ltt(delay_times,link_codes))
breathing_delays=np.asarray(breathing_reference.compute_ltt(delay_times,link_codes))

fig,axes=plt.subplots(1,2,figsize=(12,3.6))
for column,label in enumerate(link_labels):
    axes[0].plot(delay_times/86400,1e3*(selected_light_times[:,column]-selected_light_times.mean()),label=label)
axes[0].set(xlabel="mission time [days]",ylabel="delay minus six-link mean [ms]",title=f"Directed delays: {type(orbits).__name__}")
axes[0].legend(ncol=2,fontsize=7)

# Average the two directions on each geometric arm before comparing flexing.
arm_pairs=[(0,5),(1,4),(2,3)]
for first,second in arm_pairs:
    axes[1].plot(delay_times/86400,C_SI*0.5*(equal_delays[:,first]+equal_delays[:,second])/1e9,color="C0",alpha=.65)
    axes[1].plot(delay_times/86400,C_SI*0.5*(breathing_delays[:,first]+breathing_delays[:,second])/1e9,color="C3",alpha=.75)
axes[1].plot([],[],color="C0",label="equal-arm reference")
axes[1].plot([],[],color="C3",label="breathing reference")
axes[1].set(xlabel="mission time [days]",ylabel="two-way-averaged arm length [Gm]",title="Keplerian model visibly breathes")
axes[1].legend(); fig.tight_layout(); plt.show()

print(f"active six-link delay span: {1e3*np.ptp(selected_light_times):.3f} ms")
print(f"breathing-reference delay span: {1e3*np.ptp(breathing_delays):.3f} ms")"""
        ),
        md(
            r"""### 1c. Generate the one-way link data

We now generate a compact **GW-only fractional-frequency link dataset** for a
single monochromatic plane wave. For the convention used here, $i\to j$ means
emitter $i$, receiver $j$, and

$$
y^{\rm GW}_{ij}(t)=\frac12
\frac{\hat{\mathbf n}_{ij}\!\cdot
\left[\mathbf h(u_j)-\mathbf h(u_i)\right]\!\cdot
\hat{\mathbf n}_{ij}}
{1-\hat{\mathbf k}\cdot\hat{\mathbf n}_{ij}},
$$

with $u_j=t-\hat{\mathbf k}\cdot\mathbf x_j(t)/c$ and
$u_i=t-L_{ij}(t)-\hat{\mathbf k}\cdot
\mathbf x_i(t-L_{ij})/c$. The two metric samples are the reception and
emission endpoints of one laser link.

This cell does not simulate the full phasemeter budget: laser, proof-mass,
optical-path, clock, and other noises are deliberately omitted so the geometry
and delay algebra remain visible."""
        ),
        code(
            r'''TDI_DT=2.0
TDI_DURATION_DAYS=0.25
TDI_START_DAY=90.0
TDI_MARGIN=60.0  # longer than the four nested light-time delays used below
tdi_start=TDI_START_DAY*86400
tdi_duration=TDI_DURATION_DAYS*86400
tdi_time_full=np.arange(tdi_start-TDI_MARGIN,tdi_start+tdi_duration+TDI_DT,TDI_DT)
tdi_light_times=np.asarray(orbits.compute_ltt(tdi_time_full,link_codes))

GW_LINK_FREQUENCY=3e-3
GW_LINK_AMPLITUDE=1e-20
source_ra,source_dec=1.0,0.4
source_direction=np.array([
    np.cos(source_dec)*np.cos(source_ra),
    np.cos(source_dec)*np.sin(source_ra),
    np.sin(source_dec),
])
propagation_direction=-source_direction
reference_axis=np.array([0.0,0.0,1.0])
polarisation_p=np.cross(propagation_direction,reference_axis)
polarisation_p/=np.linalg.norm(polarisation_p)
polarisation_q=np.cross(propagation_direction,polarisation_p)
plus_tensor=np.outer(polarisation_p,polarisation_p)-np.outer(polarisation_q,polarisation_q)

link_data={}; link_delays={}
for column,code in enumerate(link_codes):
    emitting=int(emitter(code)); receiving=int(receiver(code))
    light_time=tdi_light_times[:,column]
    link_vector=np.asarray(orbits.compute_unit_vector(tdi_time_full,[code]))[:,0,:]
    receiver_position=np.asarray(orbits.compute_position(tdi_time_full,[receiving]))[:,0,:]
    emitter_position=np.asarray(orbits.compute_position(tdi_time_full-light_time,[emitting]))[:,0,:]
    receiver_phase=2*np.pi*GW_LINK_FREQUENCY*(
        tdi_time_full-receiver_position@propagation_direction/C_SI
    )
    emitter_phase=2*np.pi*GW_LINK_FREQUENCY*(
        tdi_time_full-light_time-emitter_position@propagation_direction/C_SI
    )
    projection=0.5*np.einsum(
        "ni,ij,nj->n",link_vector,plus_tensor,link_vector
    )/(1-link_vector@propagation_direction)
    pair=(emitting,receiving)
    link_data[pair]=GW_LINK_AMPLITUDE*projection*(
        np.cos(receiver_phase)-np.cos(emitter_phase)
    )
    link_delays[pair]=light_time

plot_window=(tdi_time_full>=tdi_start)&(tdi_time_full<tdi_start+1800)
fig,ax=plt.subplots(figsize=(10,3.6))
for pair,series in link_data.items():
    ax.plot((tdi_time_full[plot_window]-tdi_start)/60,1e22*series[plot_window],label=f"{pair[0]} to {pair[1]}")
ax.set(xlabel="minutes after data start",ylabel=r"one-way $y_{ij}^{GW}$ [$10^{-22}$]",title=f"Six GW-only link measurements: {type(orbits).__name__}")
ax.legend(ncol=3,fontsize=8); plt.show()'''
        ),
        md(
            r"""### 1d. Apply time-dependent delay operators

The basic TDI operation is not an integer array shift. Each directed link has
its own time-dependent delay:

$$
\mathcal D_{ij}a(t)=a\!\left(t-L_{ij}(t)\right).
$$

The interpolation below makes that retarded-time evaluation explicit. Nested
delays are applied one at a time; for breathing arms their order matters."""
        ),
        code(
            r'''def delay_link(series,pair):
    """Apply D_ij using the active orbit's time-dependent i-to-j delay."""
    query_time=tdi_time_full-link_delays[pair]
    finite=np.isfinite(series)
    return np.interp(
        query_time,tdi_time_full[finite],series[finite],left=np.nan,right=np.nan
    )

example_pair=(3,1)
example_return=(1,3)
link_once_delayed=delay_link(link_data[example_return],example_pair)
link_round_trip=delay_link(link_once_delayed,example_return)
fig,ax=plt.subplots(figsize=(10,3.4))
ax.plot((tdi_time_full[plot_window]-tdi_start)/60,1e22*link_data[example_return][plot_window],label="raw 1 to 3 link")
ax.plot((tdi_time_full[plot_window]-tdi_start)/60,1e22*link_once_delayed[plot_window],label="after D_31")
ax.plot((tdi_time_full[plot_window]-tdi_start)/60,1e22*link_round_trip[plot_window],label="after D_13 D_31")
ax.set(xlabel="minutes after data start",ylabel=r"fractional frequency [$10^{-22}$]",title="A delayed link is evaluated at a retarded time")
ax.legend(); plt.show()'''
        ),
        md(
            r"""### 1e. Build Michelson $X,Y,Z$

With $y_{ij}$ denoting emitter $i\to$ receiver $j$, the first-generation
unequal-arm Michelson channel centred on spacecraft 1 is

$$
\begin{aligned}
X={}&y_{31}+\mathcal D_{31}y_{13}
+\mathcal D_{31}\mathcal D_{13}y_{21}
+\mathcal D_{31}\mathcal D_{13}\mathcal D_{21}y_{12}\\
&-y_{21}-\mathcal D_{21}y_{12}
-\mathcal D_{21}\mathcal D_{12}y_{31}
-\mathcal D_{21}\mathcal D_{12}\mathcal D_{31}y_{13}.
\end{aligned}
$$

$Y$ and $Z$ follow by cycling the spacecraft indices. The code deliberately
mirrors the equation rather than hiding the delay paths in a package call."""
        ),
        code(
            r'''def michelson_xyz_channel(central,first_arm,second_arm):
    """First-generation Michelson TDI centred on `central`."""
    y=lambda emitting,receiving: link_data[(emitting,receiving)]
    d=lambda values,emitting,receiving: delay_link(values,(emitting,receiving))
    positive=(
        y(second_arm,central)
        +d(y(central,second_arm),second_arm,central)
        +d(d(y(first_arm,central),central,second_arm),second_arm,central)
        +d(d(d(y(central,first_arm),first_arm,central),central,second_arm),second_arm,central)
    )
    negative=(
        y(first_arm,central)
        +d(y(central,first_arm),first_arm,central)
        +d(d(y(second_arm,central),central,first_arm),first_arm,central)
        +d(d(d(y(central,second_arm),second_arm,central),central,first_arm),first_arm,central)
    )
    return positive-negative

X_full=michelson_xyz_channel(1,2,3)
Y_full=michelson_xyz_channel(2,3,1)
Z_full=michelson_xyz_channel(3,1,2)
tdi_keep=tdi_time_full>=tdi_start
tdi_time=tdi_time_full[tdi_keep]-tdi_start
X,Y,Z=(channel[tdi_keep] for channel in (X_full,Y_full,Z_full))
assert not any(np.isnan(channel).any() for channel in (X,Y,Z))

fig,ax=plt.subplots(figsize=(10,3.5))
show=tdi_time<3600
for channel,label in zip((X,Y,Z),("X","Y","Z")):
    ax.plot(tdi_time[show]/60,1e22*channel[show],label=label)
ax.set(xlabel="minutes after data start",ylabel=r"Michelson response [$10^{-22}$]",title="Delayed links form XYZ")
ax.legend(); plt.show()'''
        ),
        md(
            r"""### 1f. Rotate $X,Y,Z$ into $A,E,T$

Using the same orthonormal convention as JaxGB and LISA Analysis Tools,

$$
A=\frac{Z-X}{\sqrt2},\qquad
E=\frac{X-2Y+Z}{\sqrt6},\qquad
T=\frac{X+Y+Z}{\sqrt3}.
$$

This matrix rotation is orthonormal. Calling the resulting channels
statistically independent additionally requires the appropriate symmetric
$XYZ$ noise covariance; breathing unequal arms can reintroduce A/E/T cross
spectra.

For an equal-arm constellation and wavelengths long compared with the arms,
$T$ is an approximate GW-null channel. Unequal breathing arms spoil the exact
symmetry, so uncommenting `KeplerianOrbits` above and rerunning increases the
low-frequency $T$ leakage in this first-generation construction."""
        ),
        code(
            r'''A=(Z-X)/np.sqrt(2)
E=(X-2*Y+Z)/np.sqrt(6)
T=(X+Y+Z)/np.sqrt(3)

fig,axes=plt.subplots(1,2,figsize=(12,3.5))
for channel,label in zip((X,Y,Z),("X","Y","Z")):
    axes[0].plot(tdi_time[show]/60,1e22*channel[show],label=label)
for channel,label in zip((A,E,T),("A","E","T")):
    axes[1].plot(tdi_time[show]/60,1e22*channel[show],label=label)
axes[0].set(xlabel="minutes",ylabel=r"response [$10^{-22}$]",title="Michelson basis")
axes[1].set(xlabel="minutes",title="Orthogonal AET basis")
for ax in axes: ax.legend()
fig.tight_layout(); plt.show()

print(f"active orbit model: {type(orbits).__name__}")
for label,channel in zip(("A","E","T"),(A,E,T)):
    print(f"RMS {label}: {np.std(channel):.3e}")
print(f"T/A RMS ratio: {np.std(T)/np.std(A):.3e}")'''
        ),
        md(
            r""":::{admonition} TDI generation boundary
:class: warning

This transparent laboratory uses GW-only links and first-generation Michelson
TDI. It demonstrates the orbit, link, retarded-delay, XYZ, and AET data objects.
It does **not** demonstrate laser-noise cancellation for a flexing constellation:
time-dependent delay operators do not commute, and production breathing-arm
data require the correctly ordered second-generation TDI combinations.
:::"""
        ),
        md(
            """## 2. Sensitivity and Galactic confusion

As in LATW Tutorial 1, start with the noise model. The unresolved Galactic foreground changes with observing time because longer data resolve and subtract more binaries."""
        ),
        md(
            """**Predict before running:** At which frequencies should a longer observation
change the total sensitivity most: where instrumental noise dominates, or where
the unresolved Galactic foreground dominates?"""
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
ax.set(xlabel="frequency [Hz]",ylabel=r"TDI A ASD [1/$\\sqrt{\\mathrm{Hz}}$]",title="Sensitivity is part of the likelihood"); ax.legend(); plt.show()""", figure="lisa-sensitivity"
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

The following lightweight laboratory is intentionally editable. Change the
gap, drift, and noise-growth parameters and rerun it. Throughout this notebook,
a grey hatched span means **time that is unavailable to the analysis**; it is a
plot annotation, not a measured value."""
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

# One visual language for missing time everywhere in this notebook. Keep the
# plotted values masked/zero-filled as required by the demonstration, then draw
# an opaque hatched overlay so blank pixels can never be mistaken for zero
# power, clipping, or a rendering failure.
GAP_HATCH_STYLE = dict(
    facecolor="0.92", edgecolor="0.45", hatch="////", linewidth=0.0,
    alpha=1.0, zorder=20,
)


def mark_gap(axis, start=GAP_DAYS[0], end=GAP_DAYS[1], label="data gap"):
    return axis.axvspan(start, end, label=label, **GAP_HATCH_STYLE)

early=continuous_data[toy_time<7*86400]; late=continuous_data[toy_time>21*86400]
f_early,p_early=welch(early,fs=sample_rate_toy,nperseg=4096)
f_late,p_late=welch(late,fs=sample_rate_toy,nperseg=4096)
window=np.hanning(toy_time.size); fft_frequency=np.fft.rfftfreq(toy_time.size,cadence)
fft_full=np.abs(np.fft.rfft(window*continuous_data)); fft_gap=np.abs(np.fft.rfft(window*gapped_data))
f_spec,t_spec,p_spec=spectrogram(gapped_data,fs=sample_rate_toy,nperseg=2048,noverlap=1536)

fig,axes=plt.subplots(2,2,figsize=(12,7))
axes[0,0].plot(toy_time[::80]/86400,available[::80].astype(int)); axes[0,0].set(xlabel="mission time [days]",ylabel="available?",title="A three-day gap")
mark_gap(axes[0,0]); axes[0,0].legend(fontsize=8,loc="lower right")
axes[0,1].loglog(f_early[1:],np.sqrt(p_early[1:]),label="week 1"); axes[0,1].loglog(f_late[1:],np.sqrt(p_late[1:]),label="week 4")
axes[0,1].set(xlabel="frequency [Hz]",ylabel="ASD [toy]",title="Noise level changes with time"); axes[0,1].legend()
near=(fft_frequency>2.7e-3)&(fft_frequency<3.4e-3); axes[1,0].semilogy(1e3*fft_frequency[near],fft_full[near],label="continuous")
axes[1,0].semilogy(1e3*fft_frequency[near],fft_gap[near],label="gap zero-filled",alpha=.8); axes[1,0].set(xlabel="frequency [mHz]",ylabel="FFT magnitude",title="A gap spreads power across bins"); axes[1,0].legend()
band=(f_spec>2.7e-3)&(f_spec<3.4e-3); image=axes[1,1].pcolormesh(t_spec/86400,1e3*f_spec[band],np.log10(p_spec[band]+1e-30),shading="auto")
mark_gap(axes[1,1]); axes[1,1].legend(fontsize=8,loc="upper right")
axes[1,1].set(xlabel="mission time [days]",ylabel="frequency [mHz]",title="Drifting line + gap in time–frequency"); fig.colorbar(image,ax=axes[1,1],label="log power")
fig.tight_layout(); plt.show()""", figure="lisa-gap-laboratory"),
        md(
            r"""### WDM time--frequency map and likelihood

The WDM transform gives localised real coefficients $w_{nm}$ on a
time--frequency grid. For stationary Gaussian noise and a diagonal WDM
approximation,

\[
\log \mathcal L_{\rm WDM}=-\frac12\sum_{n,m}
\frac{(w^d_{nm}-w^h_{nm})^2}{\sigma_{nm}^2}+\mathrm{constant},
\qquad
\sigma_{nm}^2=\frac{N\,S(f_m)}{2\,\Delta t}.
\]

This is the WDM counterpart of the diagonal frequency-domain Whittle
likelihood. It is useful because the map makes a drifting signal, changing
noise, and a gap visible. It is not magically exact: a gap and non-stationary
noise correlate pixels, so the diagonal form below is a controlled teaching
approximation.
"""
        ),
        code(r'''toy_signal = 1.5*np.sin(phase)
WDM_NT = 280  # 28 days at 60 s cadence gives an exactly rectangular 280 x 144 grid.
wdm_data = WDM.from_time_series(WDMTimeSeries(continuous_data, dt=cadence), nt=WDM_NT)
wdm_gap = WDM.from_time_series(WDMTimeSeries(gapped_data, dt=cadence), nt=WDM_NT)
wdm_model = WDM.from_time_series(WDMTimeSeries(toy_signal, dt=cadence), nt=WDM_NT)
wdm_coeffs = np.asarray(wdm_data.coeffs[0]); wdm_gap_coeffs = np.asarray(wdm_gap.coeffs[0]); wdm_model_coeffs = np.asarray(wdm_model.coeffs[0])
wdm_nf = wdm_coeffs.shape[1]-1
# Use the transform's own grid: the row spacing is nyquist/nf = 1/(2*nf*dt),
# not 1/(nf*dt).  Building it by hand is off by a factor of two.
wdm_frequency = np.asarray(wdm_data.freq_grid)
# Unit-variance sampled white noise has one-sided PSD 2*dt.  We deliberately use
# this stationary reference even though the toy noise grows with time.
wdm_stationary_var = wdm_noise_variance(np.full(wdm_nf+1, 2*cadence), nt=WDM_NT, nf=wdm_nf, dt=cadence)
wdm_residual = wdm_coeffs-wdm_model_coeffs
logL_wdm = -.5*wdm_inner_product(wdm_residual, wdm_residual, wdm_stationary_var)
print(f"WDM grid: {WDM_NT} time pixels x {wdm_nf+1} frequency columns")
print(f"Diagonal stationary-noise WDM log likelihood (up to a constant): {logL_wdm:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
show = (wdm_frequency > 2.4e-3) & (wdm_frequency < 3.7e-3)
pixel_time = np.linspace(0, mission_days, WDM_NT)
# One shared colour scale, or the two panels cannot be compared by eye.
scale = np.percentile(np.abs(wdm_coeffs[:, show]), 99.5)
for ax, coeffs, title in zip(
    axes, (wdm_coeffs, wdm_gap_coeffs), ("continuous data", "zero-filled gap")
):
    image = ax.pcolormesh(
        pixel_time, 1e3*wdm_frequency[show], np.abs(coeffs[:, show].T),
        shading="nearest", cmap="magma", vmin=0, vmax=scale,
    )
    ax.set(xlabel="mission time [days]", ylabel="frequency [mHz]",
           title=f"WDM coefficients: {title}")
mark_gap(axes[1]); axes[1].legend(fontsize=8,loc="upper right")
fig.colorbar(image, ax=axes, label=r"$|w_{nm}|$", pad=0.02)
plt.show()

print(f"pixel size: {wdm_data.delta_t/3600:.2f} h x {wdm_data.delta_f*1e3:.3f} mHz")
print(f"line drift over the mission: {FREQUENCY_DRIFT*toy_time[-1]*1e3:.3f} mHz")
print("Raise FREQUENCY_DRIFT until the drift exceeds one pixel and the track tilts.")''', figure="lisa-wdm-map"),
        md(r"""### Frequency domain versus WDM: the same inner product

Before trusting the wavelet picture, check that it is the *same analysis*. Both
domains compute one number, the noise-weighted inner product, and an orthogonal
change of basis must leave it unchanged:

\[
(h\mid h)_{\rm freq}=4\Delta f\sum_k\frac{|\tilde h_k|^2}{S(f_k)}
\qquad\text{versus}\qquad
(h\mid h)_{\rm WDM}=\sum_{n,m}\frac{w_{nm}^2}{\sigma_{nm}^2}.
\]

If those disagree, the wavelet normalisation is wrong and nothing built on top
of it can be believed. They should agree to several decimal places."""),
        code(r'''from wdm_transform import matched_filter_snr_rfft, matched_filter_snr_wdm

# A clean, gap-free signal in stationary white noise, so both domains are valid.
check_psd_value = 2 * cadence  # one-sided PSD of unit-variance sampled noise
check_frequency = np.fft.rfftfreq(toy_time.size, cadence)

snr_frequency_domain = matched_filter_snr_rfft(
    np.fft.rfft(toy_signal),
    np.full(check_frequency.size, check_psd_value),
    check_frequency,
    dt=cadence,
)
snr_wdm_domain = matched_filter_snr_wdm(wdm_model_coeffs, wdm_stationary_var)

print(f"optimal SNR, frequency domain : {snr_frequency_domain:.5f}")
print(f"optimal SNR, WDM domain       : {snr_wdm_domain:.5f}")
print(f"ratio                         : {snr_wdm_domain/snr_frequency_domain:.6f}")
print("\nSame signal, same noise model, two different bases.")'''),
        md(r"""### What the wavelet domain actually buys you

The two domains agree on clean, stationary data, so WDM is not a better
analysis in general. It earns its place when the *assumptions* behind the
frequency-domain likelihood fail, and both failures are on display in this
laboratory:

- **A gap is local in time, but every Fourier basis function is global.** One
  missing stretch therefore corrupts every frequency bin at once, which is the
  leakage seen earlier. In WDM the corruption is confined to the pixel columns
  that overlap the gap, so it can simply be **masked**.
- **Non-stationary noise is local in time too.** The frequency-domain Whittle
  likelihood has one $S(f)$ for the whole mission. In WDM the variance carries
  a time index, $\sigma_{nm}^2$, so a drifting noise level is just a
  column-dependent weight.

The next cell puts a number on the first point: recover the same injected
signal from gapped data, once by zero-filling in the frequency domain and once
by masking pixels in WDM."""),
        code(r'''gap_pixels = (pixel_time >= GAP_DAYS[0] - wdm_data.delta_t / 86400) & (
    pixel_time <= GAP_DAYS[1] + wdm_data.delta_t / 86400
)
masked_var = wdm_stationary_var.copy()
masked_var[gap_pixels, :] = np.nan  # NaN variance drops those pixels entirely


def recovered_snr(data_coeffs, variance):
    """Matched-filter SNR of the known template against the data."""
    numerator = wdm_inner_product(wdm_model_coeffs, data_coeffs, variance)
    normalisation = np.sqrt(wdm_inner_product(wdm_model_coeffs, wdm_model_coeffs, variance))
    return numerator / normalisation


retained = available.mean()
print(f"data retained outside the gap : {retained:.3f}")
print(f"best achievable SNR           : {snr_frequency_domain*np.sqrt(retained):6.2f}")
print()
print(f"complete data                 : {recovered_snr(wdm_coeffs, wdm_stationary_var):6.2f}")
print(f"gap zero-filled, no mask      : {recovered_snr(wdm_gap_coeffs, wdm_stationary_var):6.2f}")
print(f"gap masked in WDM             : {recovered_snr(wdm_gap_coeffs, masked_var):6.2f}")
print(f"\nmasked pixel columns: {gap_pixels.sum()} of {WDM_NT}"
      f" = {gap_pixels.mean():.1%} of the mission")'''),
        md(r"""### Tracking a noise level that changes

The likelihood above deliberately used one stationary variance even though the
toy noise grows by `NOISE_GROWTH` across the mission. In the frequency domain
fixing that means segmenting the data and re-estimating a PSD per segment. In
WDM the grid is *already* segmented: each pixel column is a short stretch of
time, so the noise level per column follows from the coefficients themselves.

Estimating it from off-signal frequency rows recovers the injected growth."""),
        code(r'''# Use every frequency row above the line, so the estimate is signal-free and
# averages enough pixels to be stable.
quiet_rows = (wdm_frequency > 3.8e-3) & (wdm_frequency < 8.0e-3)
column_variance = np.nanmean(wdm_coeffs[:, quiet_rows] ** 2, axis=1)
# Each column still averages only ~70 pixels, so smooth lightly over time.
smoothing = np.ones(9) / 9
column_variance = np.convolve(column_variance, smoothing, mode="same")
column_variance[:4] = column_variance[4]
column_variance[-4:] = column_variance[-5]
# Convert pixel variance back to a noise scale relative to the stationary value.
reference = np.nanmean(wdm_stationary_var[:, quiet_rows], axis=1)
recovered_scale = np.sqrt(column_variance / reference)

injected_scale = np.interp(
    pixel_time * 86400, toy_time, noise_scale
)

fig, ax = plt.subplots(figsize=(8, 3.3))
ax.plot(pixel_time, recovered_scale, lw=1, label="recovered from WDM columns")
ax.plot(pixel_time, injected_scale, "k--", lw=2, label="injected noise scale")
ax.set(xlabel="mission time [days]", ylabel="noise level (relative)",
       title="A time-varying PSD is a column-dependent weight in WDM")
ax.legend()
plt.show()

slope = np.polyfit(pixel_time, recovered_scale, 1)[0] * mission_days
print(f"recovered growth across the mission : {slope:.3f}")
print(f"injected NOISE_GROWTH               : {NOISE_GROWTH:.3f}")'''),
        md(r"""**Where this stops being a teaching toy.** The diagonal WDM
likelihood is an approximation, not an identity. Masking whole pixel columns
throws away slightly more data than the gap itself, gap edges leave partially
contaminated pixels, and a real analysis must decide whether to mask, taper,
inpaint, or model the missing stretch, and propagate that choice into the
uncertainties. The resolution trade is also fixed by hand here: choosing `nt`
sets the pixel aspect ratio, and a signal drifting by less than one pixel
height looks stationary no matter which basis you use."""),
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
(a\mid b)=4\Delta f\,\mathrm{Re}\!\!\sum_{X\in\{A,E\},\,k}\!\!
\frac{a_{Xk}^*b_{Xk}}{S_X(f_k)},
\qquad\rho_{\rm opt}=\sqrt{(h\mid h)},
\qquad\log\mathcal L=-\tfrac12(d-h\mid d-h).
\]

- These are **exactly** the objects from the LVK notebook. Bayes' theorem does
  not change when the detector does.
- What changes: the instrument response, the source durations, the channels,
  the band, and the fact that the model is global rather than one source.
- $\Delta f=1/T_{\rm obs}$, so a longer mission means finer frequency
  resolution as well as more accumulated SNR."""
        ),
        md(
            """**Predict before running:** If two frequency templates are one Fourier bin
apart, will they be distinguishable? How should that answer change with
observation time and with SNR?"""
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
        md(
            r"""### Code studio: turn the inner product into a likelihood

Implement the Gaussian log likelihood
$\log\mathcal L=-\tfrac12(d-h\mid d-h)$ using the `inner` function above.
The checks use two limiting cases: a perfect model has zero residual, while a
zero model is worse by $\rho_{\rm opt}^2/2$."""
        ),
        code("""def student_lisa_log_likelihood(model, observed=template):
    # YOUR CODE HERE
    return None

perfect_logl = student_lisa_log_likelihood(template)
zero_logl = student_lisa_log_likelihood(np.zeros_like(template))
if perfect_logl is None or zero_logl is None:
    print("Your turn: construct the residual and return -0.5 times its inner product.")
else:
    np.testing.assert_allclose(perfect_logl, 0.0, atol=1e-10)
    np.testing.assert_allclose(zero_logl, -0.5 * optimal_snr**2, rtol=1e-10)
    print("check passed")"""),
        md(
            r"""<details>
<summary>Show one possible solution</summary>

```python
def student_lisa_log_likelihood(model, observed=template):
    residual = observed - model
    return -0.5 * inner(residual, residual)
```

</details>"""
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
        md(r"""## Extension: Fisher forecasts for LISA

Skip this section on the live route. The preceding likelihood scan contains the
main lesson; this extension explains why a one-parameter slice can look more
precise than a marginal posterior.

\[
F_{ij}=\left(\frac{\partial h}{\partial\theta_i}\Big|
\frac{\partial h}{\partial\theta_j}\right),\qquad
\sigma_i^{\rm marginal}=\sqrt{(F^{-1})_{ii}},\qquad
\sigma_i^{\rm conditional}=1/\sqrt{F_{ii}}.
\]

- Finite differences use the full moving-constellation TDI response.
- Because amplitude is a pure scaling, $\sigma_{\ln A}=1/\rho$ is an exact check.
- The scan holds phase fixed, so it should match the **conditional** error.
- Marginalising over correlated phase broadens the frequency uncertainty by
  $1/\sqrt{1-\rho_{f_0\phi_0}^2}$.
- Fisher ellipses assume high SNR and local linearity. They cannot represent
  multiple modes, hard prior edges, or strongly curved posteriors."""),
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
        md("""## 4b. Return to the live route: LISA Analysis Tools

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
            r"""## 5. The global fit: a wheel of conditional analyses

The data contain every source class and the instrument at once:

\[
d=\sum_b h_b(\theta_b)+n(\eta).
\]

The global-fit "wheel" separates that enormous problem into communicating
blocks. Each block receives a residual with the other current models removed,

\[
r_b=d-\sum_{k\ne b}h_k(\theta_k),\qquad
p(\theta_b\mid d,\theta_{-b})\propto
\pi(\theta_b)\exp\!\left[-\frac12(r_b-h_b\mid r_b-h_b)\right].
\]

{{IMAGE:assets/global_fit_wheel.png|Global Fit Wheel linking LISA source classes with instrument noise and calibration}}

*Global Fit Wheel from Katz et al.,
[Phys. Rev. D 111, 024060 (2025)](https://doi.org/10.1103/PhysRevD.111.024060),
CC BY 4.0. The black-outlined blocks were present in the initial Erebor
implementation.*

### Gibbs versus blocked Metropolis-Hastings

One **sweep** visits every block:

1. Build block $b$'s conditional residual $r_b$.
2. Update $\theta_b$ while holding the other blocks fixed.
3. Write the updated waveform/residual back to the wheel.
4. Move to the next block; repeat the wheel many times.

- **Gibbs:** draw exactly from $p(\theta_b\mid d,\theta_{-b})$ when that
  conditional distribution is available.
- **Blocked MH:** otherwise run proposals inside the block and accept/reject
  against that same conditional target.
- The blocks are not independent fits. Repeated residual exchange propagates
  uncertainty and correlations around the wheel.
- Our three-amplitude toy below has Gaussian conditional distributions, so it
  can perform genuine Gibbs draws. Real source blocks use internal MCMC/RJMCMC
  samplers, and the unknown Galactic-binary count makes the full problem
  trans-dimensional."""
        ),
        md(
            """**Predict before running:** If the first recovered source has a slightly
wrong amplitude, where does that mistake appear in the next source's fit? Why is
independent one-source-at-a-time fitting not a reliable global strategy?"""
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
            """# A one-pass subtraction is order-dependent because later blocks inherit
# the errors left by earlier ones.
sequential = np.zeros(3)
residual = data.copy()
for i, h in enumerate(templates):
    sequential[i] = global_inner(h, residual) / global_inner(h, h)
    residual -= sequential[i] * h

# Exact blocked Gibbs updates for the three amplitude coefficients. With a flat
# prior, each one-dimensional conditional posterior is Gaussian.
n_sweeps = 2000
gibbs_state = np.zeros(3)
gibbs_history = [gibbs_state.copy()]
gibbs_substates = []
gibbs_active_blocks = []
gibbs_conditional_means = []
gibbs_conditional_sds = []
for sweep in range(n_sweeps):
    for i, h in enumerate(templates):
        conditional_residual = (
            data
            - np.sum(gibbs_state[:, None, None] * templates, axis=0)
            + gibbs_state[i] * h
        )
        precision = global_inner(h, h)
        conditional_mean = global_inner(h, conditional_residual) / precision
        conditional_sd = 1 / np.sqrt(precision)
        gibbs_state[i] = rng.normal(conditional_mean, conditional_sd)
        gibbs_substates.append(gibbs_state.copy())
        gibbs_active_blocks.append(i)
        gibbs_conditional_means.append(conditional_mean)
        gibbs_conditional_sds.append(conditional_sd)
    gibbs_history.append(gibbs_state.copy())
gibbs_history = np.asarray(gibbs_history)
gibbs_substates = np.asarray(gibbs_substates)
gibbs_active_blocks = np.asarray(gibbs_active_blocks)
gibbs_conditional_means = np.asarray(gibbs_conditional_means)
gibbs_conditional_sds = np.asarray(gibbs_conditional_sds)
gibbs_samples = gibbs_history[400:]

# The simultaneous weighted least-squares solution is the Gaussian posterior
# mean, providing an independent check on the Gibbs chain.
whitened_templates=(np.sqrt(4*df/common_psd)[None,:,:]*templates).reshape(3,-1).T
whitened_data=(np.sqrt(4*df/common_psd)*data).ravel()
design=np.vstack([whitened_templates.real,whitened_templates.imag]); target=np.r_[whitened_data.real,whitened_data.imag]
joint=np.linalg.lstsq(design,target,rcond=None)[0]

print("true             ",np.round(true_scales,3))
print("one pass         ",np.round(sequential,3))
print("joint mean       ",np.round(joint,3))
print("Gibbs mean       ",np.round(gibbs_samples.mean(axis=0),3))
print("Gibbs uncertainty",np.round(gibbs_samples.std(axis=0),3))

fig,axes=plt.subplots(1,2,figsize=(11,3.6))
for i in range(3):
    axes[0].plot(gibbs_history[:80,i],lw=1,label=f"source {i+1}")
    axes[0].axhline(true_scales[i],color=f"C{i}",ls="--",alpha=.5)
    axes[1].hist(gibbs_samples[:,i],bins=32,density=True,histtype="step",color=f"C{i}",label=f"source {i+1}")
    axes[1].axvline(joint[i],color=f"C{i}",ls="--",alpha=.7)
axes[0].set(xlabel="Gibbs sweep",ylabel="amplitude multiplier",title="Conditional chains")
axes[1].set(xlabel="amplitude multiplier",ylabel="posterior density",title="Gibbs posterior")
for ax in axes: ax.legend(fontsize=8)
plt.show()"""
        ),
        md(
            r"""### Animation: one conditional block at a time

Every frame below is **one block update**, not one completed sweep. The left
panel shows the first two amplitude coordinates of the joint posterior; a
source-1 draw moves horizontally and a source-2 draw vertically. Source 3 is
not one of those axes, so its update leaves the dot fixed while still changing
the shared residual on the right.

The right panel shows what is handed to the next block. The highlighted source
is updated conditional on the current estimates of all the others, then its new
waveform is subtracted from the shared data."""
        ),
        md(
            """**Watch for:** Why can the conditional draw for one source move when
another source was updated in the preceding frame, even though the data never
changed?"""
        ),
        code(
            """# Start after a few sweeps so the narrow posterior contours remain visible.
animation_start=15
animation_stop=animation_start+36
animation_states=gibbs_substates[animation_start:animation_stop]
animation_blocks=gibbs_active_blocks[animation_start:animation_stop]
animation_means=gibbs_conditional_means[animation_start:animation_stop]
animation_sds=gibbs_conditional_sds[animation_start:animation_stop]
animation_initial=gibbs_substates[animation_start-1]
animation_path=np.vstack([animation_initial,animation_states])

amplitude_precision=np.array([
    [global_inner(first,second) for second in templates]
    for first in templates
])
amplitude_covariance=np.linalg.inv(amplitude_precision)
covariance_12=amplitude_covariance[:2,:2]
sigma_12=np.sqrt(np.diag(covariance_12))
a1_grid=np.linspace(joint[0]-4*sigma_12[0],joint[0]+4*sigma_12[0],120)
a2_grid=np.linspace(joint[1]-4*sigma_12[1],joint[1]+4*sigma_12[1],120)
A1_GRID,A2_GRID=np.meshgrid(a1_grid,a2_grid)
offsets_12=np.stack([A1_GRID-joint[0],A2_GRID-joint[1]],axis=-1)
mahalanobis_12=np.einsum("...i,ij,...j->...",offsets_12,np.linalg.inv(covariance_12),offsets_12)
posterior_12=np.exp(-.5*mahalanobis_12)

fig,(posterior_ax,residual_ax)=plt.subplots(1,2,figsize=(12,4.5))
posterior_ax.contour(A1_GRID,A2_GRID,posterior_12,
                     levels=np.exp(-.5*np.array([9.,4.,1.])),colors=".65")
posterior_ax.plot(true_scales[0],true_scales[1],"*",color="k",ms=10,label="injected")
posterior_ax.plot(joint[0],joint[1],"+",color="C3",ms=10,mew=2,label="joint mean")
gibbs_path_line,=posterior_ax.plot([],[],color=".7",lw=1)
gibbs_step_line,=posterior_ax.plot([],[],lw=3)
gibbs_point,=posterior_ax.plot([],[],"o",color="k",ms=5)
posterior_ax.set(xlabel="source 1 amplitude",ylabel="source 2 amplitude",
                 title="conditional moves through the joint posterior")
posterior_ax.legend(fontsize=8)

animation_whitening=np.sqrt(4*df/common_psd[0])
animation_data_A=np.abs(data[0])*animation_whitening
residual_ax.plot(1e3*common_frequency,animation_data_A,color=".82",lw=.8,label="shared data")
residual_line,=residual_ax.plot([],[],color="k",lw=1,label="residual after update")
active_model_line,=residual_ax.plot([],[],color=".5",lw=2,
                                    label="active source (block colour)")
conditional_text=residual_ax.text(.02,.95,"",transform=residual_ax.transAxes,va="top",
                                  bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
residual_ax.set(xlim=(1e3*common_frequency.min(),1e3*common_frequency.max()),
                ylim=(0,1.08*animation_data_A.max()),xlabel="frequency [mHz]",
                ylabel="whitened A magnitude")
residual_ax.legend(fontsize=8,loc="upper right")

def animate_gibbs_block(frame):
    state=animation_states[frame]
    previous=animation_path[frame]
    active=int(animation_blocks[frame])
    colour=f"C{active}"
    gibbs_path_line.set_data(animation_path[:frame+2,0],animation_path[:frame+2,1])
    gibbs_step_line.set_data([previous[0],state[0]],[previous[1],state[1]])
    gibbs_step_line.set_color(colour)
    gibbs_point.set_data([state[0]],[state[1]])
    residual_after=data-np.sum(state[:,None,None]*templates,axis=0)
    residual_line.set_data(1e3*common_frequency,np.abs(residual_after[0])*animation_whitening)
    active_model_line.set_data(1e3*common_frequency,
                               np.abs(state[active]*templates[active,0])*animation_whitening)
    active_model_line.set_color(colour)
    conditional_text.set_text(
        f"draw a{active+1} from its conditional\\n"
        f"mean {animation_means[frame]:.3f}, sd {animation_sds[frame]:.3f}\\n"
        f"new value {state[active]:.3f}"
    )
    posterior_ax.set_title(f"block {active+1}: condition on the other sources")
    residual_ax.set_title(f"pass this residual to block {(active+1)%3+1}")
    return (gibbs_path_line,gibbs_step_line,gibbs_point,residual_line,
            active_model_line,conditional_text)

gibbs_block_animation=FuncAnimation(fig,animate_gibbs_block,
                                    frames=len(animation_states),interval=320)
plt.close(fig); show_animation(gibbs_block_animation)"""
        ),
        md(r"""## 6. A miniature teaching-toy global fit

The three-amplitude demo above kept the source *shapes* fixed so every
conditional was exactly Gaussian. Now we drop that: two source classes with
unknown nonlinear parameters, an unknown noise level, and no idea where the
sources are until we look.

The data model is the one from the wheel, restricted to two classes:

\[
d(f)=\underbrace{A_{\rm MBHB}\,h(f;\mathcal M,t_c,\phi_c)}_{\text{one chirp}}
+\sum_{i}\underbrace{A_i\,g(f;f_{0,i},\phi_{0,i})}_{\text{monochromatic binaries}}
+\;n(f;\eta).
\]

- **Massive black-hole binary.** A restricted post-Newtonian inspiral in the
  stationary-phase approximation, sweeping upward through the band:

\[
\tilde h(f)\propto f^{-7/6}\exp\left[i\left(2\pi f t_c-\phi_c-\frac{\pi}{4}
+\frac{3}{128}(\pi\mathcal M f)^{-5/3}\right)\right].
\]

- **Galactic binaries.** Monochromatic over the observation, so each one is a
  sinc kernel of width $1/T_{\rm obs}$ centred on $f_0$. Real GBs also drift in
  frequency and are modulated by the constellation motion; Section 4 used the
  full JaxGB response for exactly that reason.
- **Noise.** A smooth analytic LISA-like PSD with one unknown overall scale
  $\eta$, so the noise level is inferred rather than assumed.

We parametrise each source by its **signal-to-noise ratio** rather than a raw
amplitude, so the numbers are directly interpretable and the truth is known.

The pipeline is the real one, in miniature:

1. **Search.** Find the sources. Nothing is known a priori.
2. **Seed.** Use the search estimates as the starting point.
3. **Gibbs.** Cycle blocks, each conditional on the current residual.

This is still an inspiral-only classroom model: it omits merger and ringdown,
the moving LISA response, gaps, and non-stationary noise. Its purpose is to
make the search--seed--cycle logic visible, not to claim a production global
fit."""),
        code(r'''from scipy.stats import gamma as gamma_dist

T_OBS_GF = 90 * 86400.0
DF_GF = 1.0 / T_OBS_GF
K_LO, K_HI = int(2.0e-3 / DF_GF), int(6.0e-3 / DF_GF)
gf_frequency = np.arange(K_LO, K_HI) * DF_GF
N_BINS = gf_frequency.size
# A smooth, analytic stand-in for the LISA noise PSD.
BASE_PSD = 2.0e-40 * (1 + (4e-4 / gf_frequency) ** 2) + 4.0e-41
MSUN_SECONDS = 4.9254909476412675e-6


def gf_inner(a, b, psd=BASE_PSD):
    return 4 * DF_GF * np.real(np.sum(np.conj(a) * b / psd))


def gf_complex_inner(a, b, psd=BASE_PSD):
    """Complex overlap; its modulus maximises over an overall phase."""
    return 4 * DF_GF * np.sum(np.conj(a) * b / psd)


def gb_shape(f0, phi0):
    """Monochromatic source observed for a finite time: a sinc kernel."""
    return np.exp(1j * phi0) * np.sinc((gf_frequency - f0) * T_OBS_GF)


def mbhb_shape(chirp_mass, t_c, phi_c):
    """Restricted post-Newtonian inspiral, stationary-phase approximation."""
    mass_seconds = chirp_mass * MSUN_SECONDS
    phase = (
        2 * np.pi * gf_frequency * t_c
        - phi_c
        - np.pi / 4
        + (3 / 128) * (np.pi * mass_seconds * gf_frequency) ** (-5 / 3)
    )
    return gf_frequency ** (-7 / 6) * np.exp(1j * phase)


def unit_norm(shape):
    return shape / np.sqrt(gf_inner(shape, shape))


def gb_template(params):
    """params = (f0, snr, phi0). The amplitude parameter *is* the SNR."""
    return params[1] * unit_norm(gb_shape(params[0], params[2]))


def mbhb_template(params):
    """params = (chirp_mass, t_c, snr, phi_c)."""
    return params[2] * unit_norm(mbhb_shape(params[0], params[1], params[3]))


TRUE_GB = np.array(
    [[3.0004e-3, 18.0, 0.7], [3.6207e-3, 25.0, 2.1], [4.7103e-3, 12.0, 4.4]]
)
TRUE_MBHB = np.array([4.0e5, 0.55 * T_OBS_GF, 60.0, 1.1])

gf_rng = np.random.default_rng(7)
injected = sum(gb_template(g) for g in TRUE_GB) + mbhb_template(TRUE_MBHB)
gf_noise = np.sqrt(BASE_PSD / (4 * DF_GF)) * (
    gf_rng.normal(size=N_BINS) + 1j * gf_rng.normal(size=N_BINS)
)
gf_data = injected + gf_noise

print(f"analysis band: {gf_frequency[0]*1e3:.1f}-{gf_frequency[-1]*1e3:.1f} mHz")
print(f"frequency bins: {N_BINS}   bin width 1/T_obs = {DF_GF:.3e} Hz")
print(f"injected total SNR: {np.sqrt(gf_inner(injected, injected)):.1f}")
print(f"noise check (n|n)/N_bins = {gf_inner(gf_noise, gf_noise)/N_BINS:.3f} (expect ~2)")'''),
        code(r'''whitened_data = np.abs(gf_data) * np.sqrt(4 * DF_GF / BASE_PSD)
whitened_mbhb = np.abs(mbhb_template(TRUE_MBHB)) * np.sqrt(4 * DF_GF / BASE_PSD)

# The magnitude panel is the object searched below, but magnitude alone throws
# away the phase evolution that makes an inspiral recognisable as a chirp.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
axes[0].plot(gf_frequency * 1e3, whitened_data, lw=0.5, color="0.6", label="data")
axes[0].plot(gf_frequency * 1e3, whitened_mbhb, lw=1.4, color="C3", label="MBHB (truth)")
for i, g in enumerate(TRUE_GB):
    axes[0].axvline(g[0] * 1e3, color="C0", ls="--", lw=1,
                    label="Galactic binaries (truth)" if i == 0 else None)
axes[0].set(xlabel="frequency [mHz]", ylabel="whitened amplitude",
            title="Search-domain data")
axes[0].legend(fontsize=8)

# The derivative of the stationary-phase Fourier phase gives the time at which
# each frequency is emitted. This is the same injected template, not a second
# waveform model or a simulated merger/ringdown.
mass_seconds = TRUE_MBHB[0] * MSUN_SECONDS
track_time = TRUE_MBHB[1] - (5 / 256) * mass_seconds * (
    np.pi * mass_seconds * gf_frequency
) ** (-8 / 3)
hours_from_tc = (track_time - TRUE_MBHB[1]) / 3600
axes[1].plot(hours_from_tc, gf_frequency * 1e3, color="C3", lw=3,
             label="MBHB inspiral (truth)")
for i, g in enumerate(TRUE_GB):
    axes[1].axhline(g[0] * 1e3, color="C0", ls="--", lw=1,
                    label="stationary GBs (truth)" if i == 0 else None)
axes[1].set(xlabel=r"time relative to $t_c$ [hours]", ylabel="frequency [mHz]",
            title="Phase reveals the rising chirp")
axes[1].legend(fontsize=8)
fig.suptitle("One inspiral chirp, three lines, and one noise level")
fig.tight_layout()
plt.show()''', figure="lisa-global-fit"),
        md(r"""### Stage 1: search

Nothing above is known to the analysis. Two searches, both reusing machinery
from earlier notebooks:

- **MBHB.** Maximise the overlap over $(\mathcal M,t_c)$. Amplitude and phase
  come out analytically, and $t_c$ enters only as $e^{2\pi i f t_c}$, so **one
  inverse FFT scans every arrival time at once**. This is exactly the
  matched-filter trick from notebook 02, Section 3.
- **Galactic binaries.** After removing the MBHB estimate, a monochromatic
  source sitting exactly on a Fourier bin has all its power in that bin, so the
  search statistic is just the **whitened periodogram**. Peaks above 7 are kept.
- **Refinement.** Real sources do not sit on bin centres, so a candidate found
  at a bin loses power to leakage. A sub-bin scan recovers it. Skipping this
  step leaves the sampler stranded in the wrong Fourier bin."""),
        code(r'''N_FFT = 1 << 17
tc_axis = np.arange(N_FFT) / (N_FFT * DF_GF)


def mbhb_search(residual, chirp_mass_grid):
    """Grid over chirp mass; one inverse FFT covers all coalescence times."""
    best = (-1.0, None, None, None)
    for chirp_mass in chirp_mass_grid:
        shape = mbhb_shape(chirp_mass, 0.0, 0.0)
        norm = np.sqrt(gf_inner(shape, shape))
        padded = np.zeros(N_FFT, dtype=complex)
        padded[K_LO:K_HI] = np.conj(residual) * shape / BASE_PSD
        overlap = 4 * DF_GF * N_FFT * np.fft.ifft(padded) / norm
        peak = int(np.argmax(np.abs(overlap)))
        if np.abs(overlap[peak]) > best[0]:
            best = (np.abs(overlap[peak]), chirp_mass, tc_axis[peak],
                    np.angle(overlap[peak]))
    return best


search_start = time.time()
snr_hat, mass_hat, tc_hat, phase_hat = mbhb_search(
    gf_data, np.geomspace(2.0e5, 8.0e5, 120)
)
snr_hat, mass_hat, tc_hat, phase_hat = mbhb_search(
    gf_data, np.geomspace(mass_hat * 0.97, mass_hat * 1.03, 60)
)
mbhb_seed = np.array([mass_hat, tc_hat, snr_hat, phase_hat])
print(f"MBHB found: SNR {snr_hat:.1f}, chirp mass {mass_hat:.4g} "
      f"(true {TRUE_MBHB[0]:.4g}), t_c/T {tc_hat/T_OBS_GF:.5f} "
      f"(true {TRUE_MBHB[1]/T_OBS_GF:.5f})")

residual_after_mbhb = gf_data - mbhb_template(mbhb_seed)
periodogram = np.abs(residual_after_mbhb) * np.sqrt(4 * DF_GF / BASE_PSD)
candidates, index = [], 1
while index < N_BINS - 1:
    if (periodogram[index] > 7.0
            and periodogram[index] >= periodogram[index - 1]
            and periodogram[index] >= periodogram[index + 1]):
        candidates.append(index)
        index += 8
    else:
        index += 1

gb_seed = []
for index in candidates:
    scan = gf_frequency[index] + np.linspace(-1.5, 1.5, 121) * DF_GF
    best = (-1.0, None)
    for f0 in scan:
        shape = unit_norm(gb_shape(f0, 0.0))
        overlap = gf_complex_inner(residual_after_mbhb, shape)
        if np.abs(overlap) > best[0]:
            best = (np.abs(overlap), [f0, np.abs(overlap), np.angle(overlap)])
    gb_seed.append(best[1])
gb_seed = np.array(gb_seed)
print(f"\nGalactic binaries found: {len(gb_seed)}  (search took "
      f"{time.time()-search_start:.1f} s)")
for i, seed in enumerate(gb_seed):
    print(f"  f0 = {seed[0]*1e3:.6f} mHz (true {TRUE_GB[i,0]*1e3:.6f}), "
          f"SNR {seed[1]:.1f} (true {TRUE_GB[i,1]:.0f})")'''),
        md(r"""### Stage 2: Gibbs, one block per source

Blocks: the MBHB, each Galactic binary, and the noise level. One sweep visits
all of them; each sees only its own conditional residual.

**Proposals matter.** The MBHB parameters are correlated and fantastically
well constrained: $t_c$ is measured to a second out of a 90-day window. An
isotropic proposal is rejected essentially always. So we build the **Fisher
matrix at the seed and propose along its eigen-directions** — the payoff for
the Fisher extension in notebook 05, Section 3, and what production
samplers actually do.

The Fisher matrix here is numerically brutal: $\sigma(f_0)\sim10^{-9}$ while
$\sigma(\rho)\sim1$, a condition number near $10^{18}$. We rescale to unit
diagonal before inverting, otherwise the amplitude errors come out meaningless.

**The noise block is a true Gibbs step.** With $\eta$ scaling the PSD,

\[
p(\eta\mid r)\propto\eta^{-N_{\rm bins}}
\exp\left[-\frac{(r\mid r)_{\rm base}}{2\eta}\right],
\]

an inverse-gamma distribution we can draw from exactly. Source blocks use
Metropolis steps — hence **Metropolis-within-Gibbs**."""),
        code(r'''def fisher_proposal(template_fn, params, steps):
    """Fisher inverse, rescaled to unit diagonal for numerical stability."""
    derivatives = []
    for i in range(len(params)):
        up, down = params.copy(), params.copy()
        up[i] += steps[i]
        down[i] -= steps[i]
        derivatives.append((template_fn(up) - template_fn(down)) / (2 * steps[i]))
    fisher = np.array([[gf_inner(a, b) for b in derivatives] for a in derivatives])
    scale = np.sqrt(np.diag(fisher))
    covariance = np.linalg.inv(fisher / np.outer(scale, scale)) / np.outer(scale, scale)
    return np.linalg.cholesky(covariance) * 2.4 / np.sqrt(len(params))


mbhb_state = mbhb_seed.copy()
gb_state = gb_seed.copy()
noise_scale = 1.0
n_sources = len(gb_state)

mbhb_jump = fisher_proposal(
    mbhb_template, mbhb_state, np.array([mbhb_state[0] * 1e-5, 0.05, 0.01, 1e-3])
)
gb_jumps = [
    fisher_proposal(gb_template, g, np.array([1e-11, 0.01, 1e-3])) for g in gb_state
]
print("MBHB Fisher sigma (chirp mass, t_c, SNR, phase):",
      np.round(np.sqrt(np.diag(mbhb_jump @ mbhb_jump.T)) * np.sqrt(4) / 2.4, 4))
print("A pure amplitude should give sigma(SNR) = 1 exactly.")'''),
        code(r'''N_SWEEPS, BURN_IN = 1800, 700
mbhb_model = mbhb_template(mbhb_state)
gb_models = [gb_template(g) for g in gb_state]

chain_mbhb = np.zeros((N_SWEEPS, 4))
chain_gb = np.zeros((N_SWEEPS, n_sources, 3))
chain_noise = np.zeros(N_SWEEPS)
accepted = np.zeros(1 + n_sources)
gibbs_rng = np.random.default_rng(11)


def scaled_chi_squared(residual, scale):
    return gf_inner(residual, residual) / scale


gibbs_start = time.time()
for sweep in range(N_SWEEPS):
    # --- block 1: the massive black-hole binary ---
    conditional = gf_data - sum(gb_models)
    current = scaled_chi_squared(conditional - mbhb_model, noise_scale)
    proposal = mbhb_state + mbhb_jump @ gibbs_rng.normal(size=4)
    if proposal[2] > 0:
        trial = mbhb_template(proposal)
        if np.log(gibbs_rng.uniform()) < -0.5 * (
            scaled_chi_squared(conditional - trial, noise_scale) - current
        ):
            mbhb_state, mbhb_model = proposal, trial
            accepted[0] += 1

    # --- blocks 2..N: one per Galactic binary ---
    for i in range(n_sources):
        others = mbhb_model + sum(gb_models[j] for j in range(n_sources) if j != i)
        conditional = gf_data - others
        current = scaled_chi_squared(conditional - gb_models[i], noise_scale)
        proposal = gb_state[i] + gb_jumps[i] @ gibbs_rng.normal(size=3)
        if proposal[1] > 0:
            trial = gb_template(proposal)
            if np.log(gibbs_rng.uniform()) < -0.5 * (
                scaled_chi_squared(conditional - trial, noise_scale) - current
            ):
                gb_state[i], gb_models[i] = proposal, trial
                accepted[1 + i] += 1

    # --- final block: the noise level, an exact inverse-gamma draw ---
    residual = gf_data - mbhb_model - sum(gb_models)
    noise_scale = (gf_inner(residual, residual) / 2) / gibbs_rng.gamma(N_BINS, 1.0)

    chain_mbhb[sweep] = mbhb_state
    chain_gb[sweep] = gb_state
    chain_noise[sweep] = noise_scale

print(f"{N_SWEEPS} sweeps in {time.time()-gibbs_start:.1f} s")
print("block acceptance rates:", np.round(accepted / N_SWEEPS, 2))'''),
        code(r'''samples_mbhb = chain_mbhb[BURN_IN:]
samples_gb = chain_gb[BURN_IN:]
samples_noise = chain_noise[BURN_IN:]


def report(name, values, truth, fmt="12.6g"):
    low, median, high = np.percentile(values, [5, 50, 95])
    flag = "ok " if low <= truth <= high else "OUT"
    print(f"  {name:12s}{median:{fmt}} [{low:{fmt}},{high:{fmt}}] "
          f"truth {truth:{fmt}}  {flag}")


print("MBHB block")
for j, name in enumerate(["chirp mass", "t_c [s]", "SNR", "phase"]):
    report(name, samples_mbhb[:, j], TRUE_MBHB[j])
print("Galactic-binary blocks")
for i in range(n_sources):
    report(f"GB{i} f0 [mHz]", samples_gb[:, i, 0] * 1e3, TRUE_GB[i, 0] * 1e3, "12.7g")
    report(f"GB{i} SNR", samples_gb[:, i, 1], TRUE_GB[i, 1])
print("Noise block")
report("PSD scale", samples_noise, 1.0)

final_residual = gf_data - mbhb_model - sum(gb_models)
print(f"\nfinal (r|r)/N_bins = "
      f"{gf_inner(final_residual, final_residual)/N_BINS:.3f}  (pure noise gives ~2)")'''),
        code(r'''fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))

axes[0, 0].plot(chain_mbhb[:, 0], lw=0.7)
axes[0, 0].axhline(TRUE_MBHB[0], color="k", ls="--")
axes[0, 0].axvspan(0, BURN_IN, color="C3", alpha=0.15)
axes[0, 0].set(xlabel="Gibbs sweep", ylabel="chirp mass", title="MBHB block")

for i in range(n_sources):
    axes[0, 1].plot(chain_gb[:, i, 1], lw=0.7, color=f"C{i}", label=f"GB{i}")
    axes[0, 1].axhline(TRUE_GB[i, 1], color=f"C{i}", ls="--", alpha=0.6)
axes[0, 1].axvspan(0, BURN_IN, color="C3", alpha=0.15)
axes[0, 1].set(xlabel="Gibbs sweep", ylabel="SNR", title="Galactic-binary blocks")
axes[0, 1].legend(fontsize=8, ncol=3)

axes[1, 0].hist(samples_noise, bins=40, density=True, histtype="step", color="C0")
axes[1, 0].axvline(1.0, color="k", ls="--")
axes[1, 0].set(xlabel=r"noise scale $\eta$", ylabel="posterior density",
               title="Noise block (exact Gibbs draws)")

axes[1, 1].plot(gf_frequency * 1e3, whitened_data, lw=0.5, color="0.7", label="data")
axes[1, 1].plot(gf_frequency * 1e3,
                np.abs(final_residual) * np.sqrt(4 * DF_GF / BASE_PSD),
                lw=0.5, color="C2", label="residual")
axes[1, 1].set(xlabel="frequency [mHz]", ylabel="whitened amplitude",
               title="All sources removed")
axes[1, 1].legend(fontsize=8)
fig.tight_layout()
plt.show()'''),
        md(r""":::{admonition} Residuals carry the history of a global fit
:class: warning

After subtracting one imperfect source, its remaining error is no longer
labelled “source 1”: it is structure in the residual. The next source block or
the noise block can absorb it, biasing their inferences. That is why global
methods repeatedly update shared residuals (or sample all blocks jointly), and
why “the residual looks quiet” is a necessary but not sufficient check.
:::
"""),
        md(r"""**Look at the GB2 trace before trusting any number.** The
weakest source (injected at SNR 12) collapses to zero amplitude early on and
stays there for several hundred sweeps before recovering. Once its amplitude
is near zero its frequency is unconstrained, so $f_0$ random-walks away and the
block has to find its way back.

- This is a genuine sampling pathology, not a plotting artefact, and it is
  common in global fits with weak sources.
- **The slowest block sets the burn-in for the whole chain.** The MBHB block
  converged within a few sweeps; GB2 needed roughly 500. Discarding a burn-in
  chosen from the MBHB trace alone would contaminate every GB2 summary.
- Production codes attack this with parallel tempering and with
  reversible-jump moves that delete and re-add sources deliberately, rather
  than waiting for a fixed-dimension chain to wander back.
- It is also the honest form of the question "is this source really there?",
  which Section 3 takes up.

What this miniature keeps from a real global fit:

- Sources are **found**, not assumed, and the fit is seeded from the search.
- Every block conditions on a residual containing the current estimate of every
  other block, so errors propagate exactly as the wheel describes.
- The noise level is inferred jointly with the signals.
- The final residual is statistically consistent with pure noise, which is the
  standard global-fit sanity check.

What it still leaves out:

- Constellation response and TDI: these are plain frequency-domain templates,
  whereas notebook 05, Section 3 used the real moving JaxGB response.
- Sky position, inclination, polarisation, and frequency drift $\dot f_0$.
- A **fixed** source count. Real analyses add and delete sources with
  reversible-jump moves inside the Galactic-binary block.
- Tens of thousands of overlapping sources rather than three, with a confusion
  foreground that is itself part of the noise model.
- Data gaps and non-stationarity, from notebook 08, Section 1."""),
        md(
            """## 7. A miniature unknown-source-count challenge

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
        md("""## Optional LISA Data Challenge input and boundary

The LDC portal requires authentication, so the live notebook uses deterministic synthetic data. An authenticated student can later upload a selected file in Colab.

```python
from google.colab import files
uploaded = files.upload()
```

This notebook uses a real sensitivity model, orbit, and TDI response. Its first
global exercise fits three amplitude coefficients from a fixed catalogue; the
second searches and samples a restricted inspiral plus three idealised Galactic
binaries with a fixed source count. Neither is a production global fit. A
research analysis must infer richer nonlinear source parameters and source
count across multiple source classes and TDI channels, jointly model the
instrument and foreground noise, include gaps and non-stationarity, and
demonstrate convergence and coverage.

Continue with:

- local `lisa_analysis_workshop/tutorials/Tutorial1.ipynb` for sensitivity/SNR/likelihood;
- Tutorial 6 for fixed-dimensional and RJ Galactic-binary inference;
- `LATW-challenge-problem.ipynb` for an MBHB plus two trans-dimensional GB groups;
- [GLASS global analysis](https://arxiv.org/abs/2301.03673);
- [LISA Data Challenge files](https://lisa-ldc.in2p3.fr/file)."""),
        md(
            """## Question bank and answer key

1. Why does a fixed LISA source acquire amplitude, phase, and frequency
   modulation during the mission, and how can that aid localisation?
2. Where should observing longer alter the total sensitivity most?
3. How do Fourier-bin separation, observing time, and SNR jointly control the
   ability to distinguish nearby frequencies?
4. Why does an imperfect source subtraction bias the rest of a global fit?
5. Why is the BIC catalogue exercise a classroom proxy rather than evidence or
   RJMCMC?
6. What is the difference between an exact Gibbs update and a blocked
   Metropolis-Hastings update?

<details>
<summary>Show the answer key</summary>

1. LISA orbits and cartwheels, changing its line of sight and delayed-link
   geometry. The resulting modulation encodes sky position and orientation.
2. The foreground-dominated part of the band changes most because longer data
   resolve and remove more Galactic binaries; instrumental noise alone is not
   reduced by resolving sources.
3. One bin, $1/T_\\mathrm{obs}$, sets an overlap scale. Longer observations make
   bins narrower, while higher SNR lets the likelihood locate a frequency to a
   fraction of that scale when the model is adequate.
4. The residual retains the first source's error, so another source/noise block
   can absorb it and become biased. Global methods repeatedly communicate via a
   shared residual or sample parameters jointly.
5. BIC uses a large-sample penalty and a fixed candidate list. It does not
   integrate the prior-weighted likelihood, explore nonlinear source parameters,
   or infer an unbounded catalogue size as a trans-dimensional method does.
6. Gibbs draws directly from a block's conditional posterior. Blocked MH uses a
   proposal and an accept/reject step that leaves the same conditional posterior
   invariant. Both must repeatedly exchange updated residuals with the other
   blocks to target the joint posterior.
</details>"""),
    ],
)


# The material above is deliberately kept in three large, reviewable source
# blocks while the course is migrated.  The public book is assembled below as
# eight independently runnable modules.  This preserves the validated teaching
# calculations while giving each topic enough room to breathe.


def _source(cell):
    return cell.get("source", "")


def _is_reference_block(cell):
    return cell.cell_type == "markdown" and _source(cell).startswith(
        "<details>\n<summary><i>Expected output"
    )


def _read_source_notebook(name):
    notebook = nbf.read(OUT / name, as_version=4)
    return [deepcopy(cell) for cell in notebook.cells if not _is_reference_block(cell)]


def _find_heading(cells, heading):
    for index, cell in enumerate(cells):
        if _source(cell).startswith(heading):
            return index
    raise ValueError(f"Could not find heading {heading!r}")


def _between(cells, start_heading, end_heading=None):
    start = _find_heading(cells, start_heading)
    end = len(cells) if end_heading is None else _find_heading(cells, end_heading)
    return [deepcopy(cell) for cell in cells[start:end]]


def _renumber(cells, mapping):
    """Restart a sliced module's section numbers at 1.

    The transitional monoliths number their sections globally, so a slice taken
    out of the middle would otherwise open at "## 6." on the website.  Rewriting
    after the slice keeps the _between() headings usable as markers.  Lettered
    subsections ("### 1a.") are left alone: they hang off a major number that
    the mapping either keeps or renames consistently.
    """
    pattern = re.compile(r"^## (\d+)([a-z]?)\.", re.MULTILINE)
    for cell in cells:
        if cell.cell_type == "markdown":
            cell.source = pattern.sub(
                lambda match: f"## {mapping.get(match.group(1), match.group(1))}"
                f"{match.group(2)}.",
                cell.source,
            )
    return cells


def _exercise(question, starter, hint, solution):
    """GWOSC-style question followed immediately by work and a hidden answer."""
    starter = re.sub(
        r"raise NotImplementedError\((.+)\)",
        r'print("Exercise ready:", \1)',
        starter,
    )
    return [
        md(f"""### Question

{question}

Write your answer in the cell immediately below. The starter runs safely before
you edit it, so the complete notebook remains reproducible."""),
        code(starter),
        md(f"""<details>
<summary>Hint</summary>

{hint}
</details>

<details>
<summary>Solution and check</summary>

```python
{solution}
```
</details>"""),
    ]


def _module_intro(goal, live_route, boundary=None):
    boundary_text = "" if boundary is None else f"\n\n**Boundary:** {boundary}"
    return md(f"""## Goal and route

{goal}

:::{'{'}admonition{'}'} Live route
:class: tip

{live_route}
:::
{boundary_text}
""")


basics_source = _read_source_notebook("00_basics_parameter_estimation.ipynb")
lvk_source = _read_source_notebook("01_lvk_compact_binary_parameter_estimation.ipynb")
lisa_source = _read_source_notebook("02_lisa_parameter_estimation_and_global_fit.ipynb")


# Part 1: keep the validated manual calculation, but replace the detached final
# question bank with questions that put a code cell directly under the task.
basics_cells = [
    _module_intro(
        "Build Bayesian parameter estimation from a signal and noise model, "
        "without hiding any step behind Bilby.",
        "Use Sections 1--4 and the PSD/Whittle bridge. Sampling algorithms and "
        "calibration are reference material.",
    ),
    *_between(basics_source, "import os", "## Question bank and answer key"),
    *_exercise(
        "Change the assumed noise standard deviation from `sigma` to "
        "`sigma / 2`. Recompute a normalized slope posterior and report how its "
        "90% interval changes. Why is a narrower posterior not automatically a "
        "better result?",
        """# Your code here
assumed_sigma = sigma / 2
raise NotImplementedError("recompute the posterior with assumed_sigma")""",
        "Reuse the `M`, `C`, `data`, and `time` arrays from the grid calculation. "
        "Only the likelihood width changes.",
        """residual = data[None, None, :] - (M[:, :, None] * time + C[:, :, None])
log_like = -0.5 * np.sum((residual / assumed_sigma) ** 2, axis=-1)
p = np.exp(log_like - np.max(log_like))
p /= np.trapezoid(np.trapezoid(p, c_grid, axis=1), m_grid)
p_m_test = np.trapezoid(p, c_grid, axis=1)
print(interval(m_grid, p_m_test))
# The interval contracts because the likelihood was made overconfident, not
# because the data became more informative.""",
    ),
]
write(
    "01_bayesian_inference.ipynb",
    "Part 1: Bayesian inference from first principles",
    basics_cells,
)


# Part 2A: signals, detector response, injection, matched filtering, and the
# manual likelihood.  Population and full sampler material move elsewhere.
lvk_setup = _between(lvk_source, "import os", "## 1. CBC parameters")
lvk_a_cells = [
    _module_intro(
        "Follow a compact-binary signal from source parameters through a detector "
        "network, injection, matched filtering, and a manual likelihood.",
        "Sections 1--4 are the core route; localisation is the extension.",
    ),
    *lvk_setup,
    *_between(lvk_source, "## 1. CBC parameters", "## 6. From events to a population"),
    *_exercise(
        "Make the template-bank offsets coarser and rerun the scan. What is the "
        "largest spacing that still places a template close to the recovered "
        "peak?",
        """# Your code here
coarse_offsets = np.linspace(-4, 4, 9)
raise NotImplementedError("call student_template_bank_scan and inspect the peak")""",
        "The relevant comparison is the grid spacing against the width of the "
        "matched-filter peak, not simply the number of templates.",
        """coarse_snr = student_template_bank_scan(coarse_offsets)
best_offset = coarse_offsets[np.argmax(coarse_snr)]
print("best coarse-grid offset:", best_offset)""",
    ),
]
write(
    "02_lvk_signals_injections.ipynb",
    "Part 2A: LVK signals, injections, and matched filtering",
    lvk_a_cells,
)


# Part 2B: a direct adaptation of GWOSC Tutorial 5.2.  The teaching path and
# numerical choices are attributed explicitly; prose and checks are adapted to
# the FQCP book and current package versions.
gwosc_bilby_cells = [
    _module_intro(
        "Analyse public H1/L1 data around GW150914 with Bilby's standard "
        "compact-binary likelihood.",
        "Build the data and PSD, inspect the priors and likelihood, then launch "
        "the sampler. The Dynesty cell can take 10--30 minutes in Colab.",
        "This is a restricted non-spinning analysis with several extrinsic "
        "parameters fixed and fast workshop sampler settings. It is not the "
        "published LVK production analysis.",
    ),
    md(r"""## Source and attribution

This notebook directly adapts **GWOSC Open Data Workshop Tutorial 5.2,
“Parameter estimation for compact object mergers.”** It preserves the
recognisable GWOSC sequence:

$$
\text{open strain}\rightarrow\text{off-source PSD}\rightarrow
\text{Bilby interferometers}\rightarrow\text{prior + likelihood}\rightarrow
\text{Dynesty}\rightarrow\text{posterior checks}.
$$

The package versions, explanations, exercises, and reproducibility checks are
adapted for this course."""),
    code("""import os, sys, subprocess, importlib.util

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
needed = ("bilby", "gwpy", "lal", "lalsimulation")
if any(importlib.util.find_spec(package) is None for package in needed):
    if IN_COLAB:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "bilby==2.8.0", "gwpy>=3.0,<4", "lalsuite==7.26.15",
        ])
    else:
        raise ImportError("Install bilby, gwpy, and lalsuite, or run in Colab.")

import numpy as np
import matplotlib.pyplot as plt
import bilby
from gwpy.timeseries import TimeSeries
from bilby.core.prior import PowerLaw, Uniform
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    generate_all_bbh_parameters,
)

bilby.core.utils.log.setup_logger(log_level="WARNING")
plt.style.use("seaborn-v0_8-whitegrid")
print("Bilby version:", bilby.__version__)"""),
    md(r"""## 1. Get the GW150914 analysis data

GW150914 occurred at GPS time $1126259462.4$. Following GWOSC, use a four-second
analysis segment with two seconds after the trigger. The 4096-Hz public product
is used explicitly."""),
    code("""time_of_event = 1126259462.4
post_trigger_duration = 2
duration = 4
sampling_frequency = 4096
analysis_start = time_of_event + post_trigger_duration - duration

analysis_data = {
    detector: TimeSeries.fetch_open_data(
        detector,
        analysis_start,
        analysis_start + duration,
        sample_rate=sampling_frequency,
        cache=True,
        timeout=120,
    )
    for detector in ("H1", "L1")
}

interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
for interferometer in interferometers:
    interferometer.minimum_frequency = 20
    interferometer.maximum_frequency = 1024
    interferometer.set_strain_data_from_gwpy_timeseries(
        analysis_data[interferometer.name]
    )

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for axis, detector in zip(axes, ("H1", "L1")):
    series = analysis_data[detector]
    axis.plot(series.times.value - time_of_event, series.value, lw=0.5)
    axis.set(ylabel=f"{detector} strain", title=f"{detector}: public 4096-Hz data")
axes[-1].set_xlabel("time from GW150914 [s]")
plt.show()""", figure="lvk-gwosc-strain"),
    md(r"""### Why does GW150914 need only four seconds?

At leading post-Newtonian order, the time remaining from a low frequency
$f_{\rm low}$ scales as

$$
t_{\rm insp}(f_{\rm low}) = \frac{5}{256}
\left(\frac{G\mathcal M}{c^3}\right)^{-5/3}
(\pi f_{\rm low})^{-8/3}.
$$

A lower-chirp-mass binary spends much longer chirping through the detector
band. CBC analyses therefore use longer data segments for lower masses and
short power-of-two segments for high-mass systems. The curve below turns the
leading-order time from 20 Hz, plus a 2.1-second post-trigger margin, into the
smallest power-of-two segment from 4 to 64 seconds.

This is an **LVK/Bilby-style teaching rule**, not a universal catalogue lookup:
production choices also depend on the low-frequency cutoff, waveform modes,
priors, and pipeline. The 4, 8, 16, 32, and 64-second analysis families are
documented for the GWTC-1 Bilby analyses in
[Romero-Shaw et al. (2020)](https://arxiv.org/abs/2006.00714)."""),
    code(r'''from scipy.constants import G, c

solar_mass_kg = 1.988409870698051e30
chirp_mass_grid = np.geomspace(2.2, 50, 500)
chirp_mass_seconds = G * chirp_mass_grid * solar_mass_kg / c**3
inspiral_time_20hz = (
    5 / 256 * chirp_mass_seconds ** (-5 / 3) * (np.pi * 20) ** (-8 / 3)
)
required_time = inspiral_time_20hz + 2.1
analysis_duration = np.clip(
    2 ** np.ceil(np.log2(required_time)), 4, 64
)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(
    chirp_mass_grid,
    required_time,
    color="0.35",
    lw=2,
    label=r"leading-order time from 20 Hz + 2.1 s",
)
ax.step(
    chirp_mass_grid,
    analysis_duration,
    where="post",
    color="C0",
    lw=3,
    label="smallest power-of-two analysis segment",
)
ax.scatter([31.2], [4], marker="*", s=180, color="C3", zorder=4)
ax.annotate(
    "GW150914",
    (31.2, 4),
    xytext=(-12, 20),
    textcoords="offset points",
    ha="right",
    arrowprops={"arrowstyle": "->", "color": "0.25"},
)
ax.set(
    xlabel=r"detector-frame chirp mass $\mathcal{M}^{\rm det}$ [$M_\odot$]",
    ylabel="duration [s]",
    title="Analysis duration follows the in-band chirp time",
    xlim=(2.2, 50),
    ylim=(0, 70),
    yticks=[4, 8, 16, 32, 64],
)
ax.set_xscale("log")
ax.set_xticks([2, 3, 5, 10, 20, 30, 50])
ax.set_xticklabels(["2", "3", "5", "10", "20", "30", "50"])
from matplotlib.ticker import NullLocator
ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(["4", "8", "16", "32", "64"])
ax.legend(frameon=False)
plt.show()''', figure="lvk-analysis-duration"),
    md(r"""### What those analysis windows contain in the time domain

The duration choice is not a statement that every second contains signal. The
examples below show three **equal-mass** binaries in their selected records,
recentered on merger for display as in the NZ workshop (only the first 0.35 s
after merger is shown). The 60-$M_\odot$ system enters at 20 Hz only near the
end of its four-second record; the 6-$M_\odot$ system has a long, slow inspiral,
so it needs a much longer record.

These use the same **IMRPhenomPv2** inspiral--merger--ringdown model as the
NZ workshop, rendered through Bilby/LAL. Each panel has its own amplitude scale
and uses equal masses solely to make the duration comparison transparent; it is
not an event reconstruction."""),
    code(r'''# Equal-mass IMRPhenomPv2 signals in their selected analysis windows.
# The physical waveform removes the artificial ISCO cut in a Newtonian cartoon.
def normalized_imr_signal(total_mass, window_duration, sample_rate=2048):
    chirp_mass = total_mass * 0.25 ** (3 / 5)
    generator = bilby.gw.WaveformGenerator(
        duration=window_duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2",
            "reference_frequency": 20.0,
            "minimum_frequency": 20.0,
        },
    )
    parameters = dict(
        mass_1=total_mass / 2,
        mass_2=total_mass / 2,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_jl=0.0,
        phi_12=0.0,
        lambda_1=0.0,
        lambda_2=0.0,
        luminosity_distance=1000.0,
        theta_jn=0.0,
        ra=0.0,
        dec=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=0.0,
    )
    strain = generator.time_domain_strain(parameters)["plus"]
    # The frequency-domain model is represented on a periodic FFT grid.  Roll
    # the peak into the interior, as in the NZ workshop, before plotting it.
    strain = np.roll(strain, -len(strain) // 3)
    strain /= np.max(np.abs(strain))
    peak = np.argmax(np.abs(strain))
    time_from_merger = generator.time_array - generator.time_array[peak]
    return time_from_merger, strain, chirp_mass

examples = [(6, 64), (20, 8), (60, 4)]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
for ax, (total_mass, window_duration) in zip(axes, examples):
    time, strain, chirp_mass = normalized_imr_signal(
        total_mass, window_duration
    )
    ax.plot(time, strain, color="C0", lw=0.8)
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set(
        xlim=(-2 * window_duration / 3, min(window_duration / 3, 0.35)),
        ylim=(-1.12, 1.12),
        xlabel="time from merger [s]",
        title=(
            rf"$M_{{\rm tot}}={total_mass}\,M_\odot$; "
            rf"$\mathcal{{M}}={chirp_mass:.1f}\,M_\odot$; "
            rf"{window_duration}-s record"
        ),
    )
    ax.set_yticks([])
axes[0].set_ylabel("normalised strain\n(arbitrary amplitude per panel)")
fig.suptitle("IMRPhenomPv2 signal within the selected analysis window", y=1.03)
plt.tight_layout()
plt.show()''', figure="lvk-analysis-time-domain"),
    md(r"""## 2. Estimate the PSD from off-source data

The likelihood weights residuals by a noise PSD. Following the GWOSC tutorial,
estimate it from the 128 seconds immediately before the analysis segment using
four-second Tukey-windowed periodograms, 50% overlap, and a median average. This
is deliberately off source: the event should not define its own noise weighting.

This common off-source construction is usually called a **median-Welch PSD**:

1. **Why Welch?** A single periodogram is an extremely noisy estimate: at each
   frequency its scatter is comparable to its mean. Split a long record into
   short, windowed stretches; Fourier transform each; then combine their
   periodograms. We trade frequency resolution for much lower estimator
   variance.
2. **Why a median rather than an arithmetic mean?** For perfectly stationary
   Gaussian noise the mean is efficient. Real interferometer data contain
   short glitches, and one loud stretch can pull a mean far upward. The median
   is robust to a minority of contaminated stretches. The implementation
   applies the appropriate median-bias correction; the median is not taken as
   an already unbiased PSD by itself.
3. **Why overlap?** A Tukey or Hann window downweights samples near each
   stretch's edges. Sliding the next stretch by half a window reuses those
   samples near another window's centre and supplies more periodograms from a
   fixed off-source record. Adjacent estimates are correlated, so 50% overlap
   improves stability but does **not** double the independent information.

Keep the four timescales distinct:

| quantity | value here | role |
|---|---:|---|
| event-analysis duration | 4 s | data entering the likelihood |
| Welch FFT length | 4 s | one periodogram; gives 0.25-Hz bins |
| overlap / stride | 2 s / 2 s | 50% overlap between periodograms |
| off-source PSD record | 128 s | reservoir supplying 63 stretches |

Median-Welch is common, not the only defensible LVK noise model. BayesLine-like
on-source fits and pipeline-specific PSD estimators answer related but different
questions; see [Chatziioannou et al. (2019)](https://arxiv.org/abs/1907.06540)."""),
    code("""# A visual Welch construction: one transient contaminates a few windows,
# while the median remains representative of the many quiet windows.
from scipy.signal import get_window, periodogram, welch

welch_rng = np.random.default_rng(2026)
toy_rate = 256
toy_duration = 128
fft_length = 4
nperseg = toy_rate * fft_length
noverlap = nperseg // 2
toy_time = np.arange(toy_rate * toy_duration) / toy_rate

# Smooth coloured noise plus one deliberately obvious broadband transient.
toy_frequency = np.fft.rfftfreq(toy_time.size, 1 / toy_rate)
toy_shape = 1 + (18 / np.maximum(toy_frequency, 1)) ** 4
quiet_noise = np.fft.irfft(
    np.fft.rfft(welch_rng.normal(size=toy_time.size)) * np.sqrt(toy_shape),
    n=toy_time.size,
)
glitch = (
    100
    * np.exp(-0.5 * ((toy_time - 11.2) / 0.08) ** 2)
    * np.sin(2 * np.pi * 70 * (toy_time - 11.2))
)
contaminated_noise = quiet_noise + glitch

starts = np.arange(0, toy_time.size - nperseg + 1, nperseg - noverlap)
window = get_window(("tukey", 0.2), nperseg)
segment_psds = []
touches_glitch = []
for start in starts:
    frequency, one_psd = periodogram(
        contaminated_noise[start : start + nperseg],
        fs=toy_rate,
        window=window,
        scaling="density",
    )
    segment_psds.append(one_psd)
    touches_glitch.append(start / toy_rate <= 11.2 < (start + nperseg) / toy_rate)
segment_psds = np.asarray(segment_psds)

_, clean_reference = welch(
    quiet_noise, fs=toy_rate, window=("tukey", 0.2),
    nperseg=nperseg, noverlap=noverlap, average="median",
)
_, contaminated_mean = welch(
    contaminated_noise, fs=toy_rate, window=("tukey", 0.2),
    nperseg=nperseg, noverlap=noverlap, average="mean",
)
_, contaminated_median = welch(
    contaminated_noise, fs=toy_rate, window=("tukey", 0.2),
    nperseg=nperseg, noverlap=noverlap, average="median",
)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
shown = toy_time < 16
axes[0].plot(toy_time[shown], contaminated_noise[shown], color="0.25", lw=0.7)
for index, start in enumerate(starts[:7]):
    left = start / toy_rate
    axes[0].axvspan(left, left + fft_length, color=f"C{index % 2}", alpha=0.10)
axes[0].axvline(11.2, color="C3", ls="--", label="transient")
axes[0].set(xlabel="off-source time [s]", ylabel="toy strain",
            title="4-s windows, shifted by 2 s")
axes[0].legend(frameon=False)

band = (frequency >= 15) & (frequency <= 120)
for one_psd, bad in zip(segment_psds, touches_glitch):
    axes[1].loglog(
        frequency[band], np.sqrt(one_psd[band]),
        color="C3" if bad else "0.7", alpha=0.85 if bad else 0.16,
        lw=1.1 if bad else 0.7,
    )
axes[1].set(xlabel="frequency [Hz]", ylabel="ASD [toy units]",
            title="Each window gives a noisy periodogram")

axes[2].loglog(frequency[band], np.sqrt(clean_reference[band]),
               color="0.2", ls="--", lw=2, label="quiet reference")
axes[2].loglog(frequency[band], np.sqrt(contaminated_mean[band]),
               color="C3", alpha=0.85, label="mean with transient")
axes[2].loglog(frequency[band], np.sqrt(contaminated_median[band]),
               color="C0", lw=2, label="median with transient")
axes[2].set(xlabel="frequency [Hz]", ylabel="ASD [toy units]",
            title=f"Combine {len(starts)} overlapping estimates")
axes[2].legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.show()""", figure="lvk-welch-explainer"),
    code("""psd_duration = 32 * duration
psd_start = analysis_start - psd_duration

psd_data = {
    detector: TimeSeries.fetch_open_data(
        detector,
        psd_start,
        psd_start + psd_duration,
        sample_rate=sampling_frequency,
        cache=True,
        timeout=120,
    )
    for detector in ("H1", "L1")
}

for interferometer in interferometers:
    alpha = 2 * interferometer.strain_data.roll_off / duration
    estimate = psd_data[interferometer.name].psd(
        fftlength=duration,
        overlap=duration / 2,
        window=("tukey", alpha),
        method="median",
    )
    interferometer.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=estimate.frequencies.value,
        psd_array=estimate.value,
    )

fig, ax = plt.subplots(figsize=(9, 4))
for interferometer in interferometers:
    mask = interferometer.strain_data.frequency_mask
    frequency = interferometer.frequency_array[mask]
    asd = interferometer.amplitude_spectral_density_array[mask]
    ax.loglog(frequency, asd, label=interferometer.name)
ax.set(xlabel="frequency [Hz]", ylabel=r"ASD [1/$\\sqrt{\\mathrm{Hz}}$]",
       title="Off-source noise model used by the likelihood")
ax.legend()
plt.show()""", figure="lvk-gwosc-psd"),
    *_exercise(
        "Why should the PSD normally be estimated away from the signal? Compute "
        "the ratio of the H1 ASD near 100 Hz to its median between 80 and 120 Hz "
        "as a quick local sanity check.",
        """# Your code here
h1 = interferometers[0]
raise NotImplementedError("select the 80--120 Hz bins and compare ASD values")""",
        "Use `h1.frequency_array` and `h1.amplitude_spectral_density_array`; "
        "apply the interferometer's frequency mask as well.",
        """mask = h1.strain_data.frequency_mask & (h1.frequency_array >= 80) & (h1.frequency_array <= 120)
band_frequency = h1.frequency_array[mask]
band_asd = h1.amplitude_spectral_density_array[mask]
near_100 = np.argmin(np.abs(band_frequency - 100))
print(band_asd[near_100] / np.median(band_asd))""",
    ),
    md(r"""## 3. Define the restricted prior

Sample detector-frame chirp mass, mass ratio, phase, coalescence time, and
luminosity distance. Fix spins, sky position, orientation, and polarisation to
keep the workshop runtime manageable. The prior is part of this analysis; these
fixed values are not facts established by the data."""),
    code("""prior = bilby.core.prior.PriorDict()
prior["chirp_mass"] = Uniform(30.0, 32.5, name="chirp_mass")
prior["mass_ratio"] = Uniform(0.5, 1.0, name="mass_ratio")
prior["phase"] = Uniform(0, 2 * np.pi, name="phase", boundary="periodic")
prior["geocent_time"] = Uniform(
    time_of_event - 0.1, time_of_event + 0.1, name="geocent_time"
)
prior["luminosity_distance"] = PowerLaw(
    alpha=2, minimum=50, maximum=2000,
    name="luminosity_distance", unit="Mpc"
)
prior.update({
    "a_1": 0.0, "a_2": 0.0, "tilt_1": 0.0, "tilt_2": 0.0,
    "phi_12": 0.0, "phi_jl": 0.0, "dec": -1.2232,
    "ra": 2.19432, "theta_jn": 1.89694, "psi": 0.532268,
})
prior"""),
    md(r"""## 4. Build the waveform and likelihood

`WaveformGenerator` maps parameters to polarizations. Each interferometer then
applies its response and arrival-time delay. `GravitationalWaveTransient`
evaluates the coherent PSD-weighted residual likelihood. Time, phase, and
distance are analytically marginalised during sampling and reconstructed later."""),
    code("""waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=20.0,
)
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
    priors=prior,
    time_marginalization=True,
    phase_marginalization=True,
    distance_marginalization=True,
)
print("Likelihood:", type(likelihood).__name__)
print("Frequency bins used:", sum(ifo.strain_data.frequency_mask.sum() for ifo in interferometers))"""),
    md("""## 5. Run Dynesty

This is the genuine sampler call from the GWOSC workflow. The deliberately
loose stopping criterion and small live-point count reduce classroom runtime.
Do not reuse them as production defaults."""),
    code("""result_short = bilby.run_sampler(
    likelihood=likelihood,
    priors=prior,
    sampler="dynesty",
    outdir="gw150914_short",
    label="GW150914",
    conversion_function=generate_all_bbh_parameters,
    nlive=250,
    dlogz=1.0,
    sample="rwalk",
    clean=False,
)

parameters = ["chirp_mass", "mass_ratio", "geocent_time", "phase"]
print(result_short.posterior[parameters].describe(percentiles=[0.05, 0.5, 0.95]))
print(f"log Bayes factor: {result_short.log_bayes_factor:.1f} +/- {result_short.log_evidence_err:.1f}")
result_short.plot_corner(parameters=parameters, prior=True, save=False)
plt.show()""", figure="lvk-gwosc-bilby-corner"),
    *_exercise(
        "Report the median and symmetric 90% credible interval for detector-frame "
        "chirp mass. Then state one reason it should not match every published "
        "GW150914 posterior exactly.",
        """# Your code here
chirp_mass_samples = result_short.posterior["chirp_mass"].to_numpy()
raise NotImplementedError("calculate the 5th, 50th, and 95th percentiles")""",
        "Use `np.quantile(chirp_mass_samples, [0.05, 0.5, 0.95])`. The waveform, "
        "fixed spins/extrinsic parameters, calibration treatment, PSD, priors, "
        "and sampler settings all define the comparison.",
        """low, median, high = np.quantile(chirp_mass_samples, [0.05, 0.5, 0.95])
print(f"Mc_det = {median:.2f} [{low:.2f}, {high:.2f}] Msun")
print("This restricted non-spinning workshop model is not the published full analysis.")""",
    ),
]
write(
    "03_lvk_gw150914_bilby.ipynb",
    "Part 2B: GW150914 with Bilby",
    gwosc_bilby_cells,
)


# Part 2C: the compact population demonstration becomes an independent module
# and gains local exercises/checks.
population_setup = code("""import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(20260817)
plt.style.use("seaborn-v0_8-whitegrid")""")
lvk_c_cells = [
    _module_intro(
        "Turn event-level information into a population statement and see why "
        "the detected catalogue is not the underlying population.",
        "Run the selection-bias laboratory, then complete the question cell.",
        "The toy treats event masses as exactly measured. Production population "
        "inference reweights uncertain event posterior samples and estimates "
        "selection with injection campaigns.",
    ),
    population_setup,
    *_renumber(_between(lvk_source, "## 6. From events to a population"), {"6": "1"}),
    md(r"""## 2. Events are posteriors, not numbers

Section 1 treated every mass as exactly measured so that selection bias stayed
visible on its own. Real events arrive as **posterior samples**, and that
changes the hierarchical likelihood: the single-event term becomes an average
over that event's samples,

$$
\int p(d_i\mid\theta)\,p(\theta\mid\Lambda)\,d\theta
\;\propto\;
\frac{1}{S}\sum_{s=1}^{S}
\frac{p(\theta_{is}\mid\Lambda)}{\pi_{\rm PE}(\theta_{is})}.
$$

The division by $\pi_{\rm PE}$ is the part that gets forgotten. Posterior
samples were drawn under **the prior the PE run used**, not under a flat prior.
If you average $p(\theta\mid\Lambda)$ over them without dividing that prior out,
the population silently inherits the shape of the PE prior.

A second factor changes too. Detection depended on the *true* mass, so
$p_{\rm det}$ belongs inside the per-event integral:

$$
\frac{1}{S}\sum_s
\frac{p(\theta_{is}\mid\Lambda)\,p_{\rm det}(\theta_{is})}{\pi_{\rm PE}(\theta_{is})}
\Bigg/ \alpha(\Lambda).
$$

In Section 1 the masses were exact, so $p_{\rm det}(m_i)$ was a constant that
cancelled out of the hyper-posterior. The moment the masses became uncertain it
stopped cancelling. Nothing warns you about this: the code from Section 1 keeps
running perfectly well and returns a slightly wrong answer.

Below, the PE runs used $\pi_{\rm PE}(m)\propto m^4$. A rising prior like this
is an ordinary choice — priors flat in distance-cubed or in detector-frame
mass do the same thing — and it is steep here so the effect is visible with
only 40 events."""),
    md("""**Predict before running:** the PE prior rises with mass. If it is left
in, should the inferred population mean come out too high or too low?"""),
    code(
        """measurement_sigma = 3.0
n_event_samples = 400


def pe_prior(mass):
    return mass**4


# Each detected event is measured with finite precision, then analysed under
# pe_prior.  Importance resampling turns the measurement likelihood into
# posterior samples without needing a sampler here.
noisy_observation = detected + rng.normal(0, measurement_sigma, detected.size)
proposal = np.clip(
    rng.normal(noisy_observation[:, None], measurement_sigma, (detected.size, 4000)),
    8,
    55,
)
proposal_weights = pe_prior(proposal)
proposal_weights /= proposal_weights.sum(axis=1, keepdims=True)
event_samples = np.array(
    [
        rng.choice(row, size=n_event_samples, p=weight)
        for row, weight in zip(proposal, proposal_weights)
    ]
)
print("posterior samples per event:", event_samples.shape)""",
    ),
    code(
        """def hyper_posterior(divide_pe_prior, samples=None, grid=None):
    samples = event_samples if samples is None else samples
    grid = mean_grid if grid is None else grid
    values = []
    for mean in grid:
        population_density = norm.pdf(samples, mean, population_width)
        if divide_pe_prior:
            population_density = population_density / pe_prior(samples)
        # Detection depended on the true mass, which is now uncertain, so the
        # selection function sits inside the per-event integral as well.  In
        # Section 1 the masses were exact and this factor cancelled.
        population_density = population_density * detection_probability(samples)
        alpha = np.trapezoid(
            norm.pdf(integration_grid, mean, population_width)
            * detection_probability(integration_grid),
            integration_grid,
        )
        values.append(
            np.log(population_density.mean(axis=1)).sum()
            - len(samples) * np.log(alpha)
        )
    return np.array(values)


correct = normalise_population(hyper_posterior(divide_pe_prior=True))
prior_contaminated = normalise_population(hyper_posterior(divide_pe_prior=False))

fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(mean_grid, correct, color="C0", lw=2, label="PE prior divided out")
ax.plot(mean_grid, prior_contaminated, color="C3", lw=2, label="PE prior left in")
ax.axvline(population_mean, color="k", ls="--", label="injection")
ax.set(
    xlabel="population mean",
    ylabel="posterior density",
    title="Same data, same sampler, one forgotten prior",
)
ax.legend(fontsize=8)
plt.show()

correct_map = mean_grid[np.argmax(correct)]
contaminated_map = mean_grid[np.argmax(prior_contaminated)]
print(f"injected population mean : {population_mean:.2f}")
print(f"PE prior divided out     : {correct_map:.2f}")
print(f"PE prior left in         : {contaminated_map:.2f}")
print(f"shift                    : {contaminated_map - correct_map:+.2f}")""",
        figure="population-pe-prior",
    ),
    md("""**Do not read the single-catalogue numbers as a verdict.** With 40
events the statistical scatter on the population mean is of order one mass unit,
so on any one realisation either curve can land closer to the injection by
luck — including the wrong one. That is precisely why a bug like this survives
code review.

The way to see it is to repeat the experiment. The scatter averages away; a
systematic error does not."""),
    code(
        """coarse_grid = np.linspace(18, 38, 80)
shifts, correct_errors = [], []
for trial in range(20):
    trial_rng = np.random.default_rng(500 + trial)
    trial_detected = all_masses[
        trial_rng.random(all_masses.size) < detection_probability(all_masses)
    ][:40]
    trial_observed = trial_detected + trial_rng.normal(
        0, measurement_sigma, trial_detected.size
    )
    trial_proposal = np.clip(
        trial_rng.normal(
            trial_observed[:, None], measurement_sigma, (trial_detected.size, 4000)
        ),
        8,
        55,
    )
    trial_weights = pe_prior(trial_proposal)
    trial_weights /= trial_weights.sum(axis=1, keepdims=True)
    trial_samples = np.array(
        [
            trial_rng.choice(row, size=n_event_samples, p=weight)
            for row, weight in zip(trial_proposal, trial_weights)
        ]
    )
    good = coarse_grid[np.argmax(hyper_posterior(True, trial_samples, coarse_grid))]
    bad = coarse_grid[np.argmax(hyper_posterior(False, trial_samples, coarse_grid))]
    shifts.append(bad - good)
    correct_errors.append(good - population_mean)

shifts = np.array(shifts)
correct_errors = np.array(correct_errors)
print(f"correct analysis, error over 20 catalogues : "
      f"{correct_errors.mean():+.2f} +/- {correct_errors.std():.2f}")
print(f"shift from the forgotten prior             : "
      f"{shifts.mean():+.2f} +/- {shifts.std():.2f}")
print(f"shifts with the same sign                  : "
      f"{(shifts > 0).sum()}/20")
assert (shifts > 0).sum() >= 18, "the forgotten prior should bias one way"
print("check passed: the error is systematic, not scatter")""",
    ),
    md("""The correct analysis scatters around the injection, with a residual offset
well inside its own spread — that is what 40 events buys you. The forgotten
prior shifts the answer the *same way every time* —
that is the signature of a systematic, and it is why repeated simulation, not a
single convincing-looking run, is what validates a population pipeline.

Nothing about the sampler was wrong here. No convergence diagnostic would flag
it, and the contaminated posterior looks perfectly healthy on its own. This is
the characteristic failure of hierarchical inference: **the bug lives in the
bookkeeping between stages, not inside either stage.**"""),
    md("""## Checklist: what to verify before a population claim

Each item below has bitten a real analysis.

| Check | What it catches |
| --- | --- |
| sampler convergence and $N_{\\rm eff}$ at **event** level | a hyper-posterior built on unconverged event runs |
| posterior samples pressing against prior boundaries | an event whose prior, not data, set its width |
| posterior-predictive and residual checks per event | waveform or noise mismodelling entering the population |
| **dividing out the event-level sampling prior** | the bias demonstrated above |
| enough samples per event that the Monte Carlo sum is stable | a hyper-likelihood dominated by a handful of samples |
| injections that match the analysed population **and** pipeline | a selection function $\\alpha(\\Lambda)$ that describes a different search |
| repeating under alternative waveform, calibration, and population models | a result that is really a statement about one model choice |

The last one is the difference between "the data prefer this population" and
"this population model, fitted to these data, prefers these hyperparameters"."""),
]
write(
    "04_lvk_population_and_checks.ipynb",
    "Part 2C: LVK populations and PE checks",
    lvk_c_cells,
)


# Part 3A: response and signal foundations, including the manual likelihood and
# AnalysisContainer bridge.  WDM and global fitting are moved to their own files.
lisa_setup = _between(lisa_source, "import os", "## 1. LISA's band and source zoo")
lisa_a_cells = [
    _module_intro(
        "Connect LISA's source zoo to its moving constellation, delayed links, "
        "TDI variables, sensitivity, and likelihood interfaces.",
        "Follow orbit -> links -> XYZ/AET, then the inner-product likelihood. "
        "The Fisher section is optional.",
    ),
    *lisa_setup,
    *_between(lisa_source, "## 1. LISA's band and source zoo", "## 3. Real LISA data will be more complicated"),
    *_renumber(_between(lisa_source, "## 4. Inner product, SNR, and likelihood", "## 5. The global fit: a wheel of conditional analyses"), {"4": "3"}),
    md("""## Analysis-code map

| Tool family | Typical role in a LISA analysis |
| --- | --- |
| LISA Analysis Tools / `lisatools` | sensitivity, response containers, likelihood plumbing |
| JaxGB | fast Galactic-binary response generation |
| GLASS / Erebor-style pipelines | overlapping-source and global-fit workflows |
| Eryn | ensemble and trans-dimensional sampling machinery |
| Gemoo-style searches | source search/proposal construction in crowded data |

These packages are not interchangeable single commands. Each occupies a layer
between waveform/response generation, search, likelihood construction, and
posterior exploration. The following notebooks isolate the global-fit, PSD,
and time-frequency layers."""),
    *_exercise(
        "Set `USE_BREATHING_ORBITS = True`, rerun the orbit-to-TDI chain, and "
        "compare the RMS of T against A. Why is T an approximate rather than "
        "universal null channel?",
        """# Change the switch near the orbit construction, then rerun Sections 1a--1f.
USE_BREATHING_ORBITS = True
raise NotImplementedError("report the recomputed T/A RMS ratio")""",
        "The cancellation depends on frequency, arm equality, response details, "
        "and which physical contribution is being considered.",
        """print("T/A RMS ratio:", np.std(T) / np.std(A))
print("Interpret this for the simulated link model only, not as a universal null.")""",
    ),
]
write(
    "05_lisa_signals_response_codes.ipynb",
    "Part 3A: LISA signals, response, and analysis codes",
    lisa_a_cells,
)


global_lisa_state = code("""orbits = EqualArmlengthOrbits()
t_obs = 90 * 86400.0
df = 1.0 / t_obs
simulator = JaxGB(orbits, t_obs=t_obs, t0=0, n=128)
print("Response model:", type(orbits).__name__)
print("Observation time [days]:", t_obs / 86400)""")
lisa_b_cells = [
    _module_intro(
        "Understand a global fit as communicating conditional analyses through "
        "a shared residual, then implement exact and Metropolis-within-Gibbs blocks.",
        "Run the fixed-shape demonstration and the search -> seed -> cycle toy.",
        "The examples use fixed source counts or BIC enumeration and simplified "
        "responses. They are not production trans-dimensional LISA inference.",
    ),
    *lisa_setup,
    global_lisa_state,
    *_renumber(_between(lisa_source, "## 5. The global fit: a wheel of conditional analyses", "## Optional LISA Data Challenge input and boundary"), {"5": "1", "6": "2", "7": "3"}),
    *_exercise(
        "Reverse the order of the one-pass source subtraction. Compare the final "
        "residual norm with the blocked chain. Why can one-pass subtraction "
        "depend on ordering?",
        """# Your code here
reverse_order = [2, 1, 0]
raise NotImplementedError("repeat the one-pass updates in reverse_order")""",
        "Each early subtraction is conditional on an incomplete residual. Its "
        "error is inherited by later blocks and is never revisited.",
        """reverse_residual = data.copy()
reverse_amplitudes = np.zeros(3)
for source_index in reverse_order:
    template_i = templates[source_index]
    reverse_amplitudes[source_index] = global_inner(template_i, reverse_residual) / global_inner(template_i, template_i)
    reverse_residual -= reverse_amplitudes[source_index] * template_i
print("reverse one-pass residual norm:", global_inner(reverse_residual, reverse_residual))""",
    ),
]
write(
    "06_lisa_global_fit_gibbs.ipynb",
    "Part 3B: LISA global fitting and Gibbs sampling",
    lisa_b_cells,
)


# Part 3C: a compact, self-contained P-spline PSD laboratory.  It uses the
# Whittle objective directly and a Laplace approximation only for a visual
# uncertainty band; that approximation is labelled rather than promoted as a
# production posterior.
lisa_c_cells = [
    _module_intro(
        "Estimate a smooth PSD with a cubic P-spline inside a Whittle likelihood "
        "and see how the roughness penalty changes the answer.",
        "Fit the simulated spectrum, inspect posterior-predictive whitening, then "
        "change the smoothing strength in the question cell.",
        "The coefficient band uses a local Laplace approximation. A production "
        "Bayesian P-spline analysis must sample the full posterior and validate "
        "coverage, convergence, line treatment, and foreground identifiability.",
    ),
    code("""import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import minimize

rng = np.random.default_rng(20260817)
plt.style.use("seaborn-v0_8-whitegrid")"""),
    md(r"""## 1. Why estimate the PSD inside the analysis?

For approximately independent complex Fourier coefficients $d_k$ with
one-sided PSD $S_k$, the Whittle negative log likelihood is, up to constants,

$$
-\log\mathcal L = \sum_k\left[\log S_k + \frac{I_k}{S_k}\right],
\qquad I_k=|d_k|^2.
$$

A raw periodogram is too variable to use as a smooth noise model.

There are many different models for PSDs. In this notebook we use *penalised*
splines. Write $\log S(f)=B(f)\beta$ in a cubic B-spline basis and penalise
second differences,

$$
-\log p(\beta\mid I)= -\log\mathcal L
+\frac{\lambda}{2}\lVert D_2\beta\rVert^2 + \mathrm{constant}.
$$

![A penalised spline built from weighted basis functions](https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/assets/pspline_explainer.gif)"""),
    code("""n_frequency = 1200
frequency = np.geomspace(2e-4, 2e-2, n_frequency)
log_frequency = np.log(frequency)

true_psd = 2.5e-40 * (
    1 + (8e-4 / frequency) ** 4 + 0.12 * (frequency / 7e-3) ** 2
)
true_psd *= 1 + 1.8 * np.exp(-0.5 * ((np.log10(frequency) + 2.55) / 0.09) ** 2)
periodogram = true_psd * rng.exponential(size=n_frequency)

degree = 3
n_basis = 24
interior = np.linspace(log_frequency.min(), log_frequency.max(), n_basis - degree + 1)
knots = np.r_[[interior[0]] * degree, interior, [interior[-1]] * degree]
basis = np.column_stack([
    BSpline(knots, np.eye(n_basis)[index], degree)(log_frequency)
    for index in range(n_basis)
])
second_difference = np.diff(np.eye(n_basis), n=2, axis=0)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.loglog(frequency, periodogram, color="0.75", lw=0.6, label="periodogram")
ax.loglog(frequency, true_psd, "k--", lw=2, label="injected PSD")
ax.set(xlabel="frequency [Hz]", ylabel="PSD", title="A periodogram is not a smooth PSD estimate")
ax.legend()
plt.show()""", figure="lisa-pspline-periodogram"),
    md("""## 2. Fit the penalised Whittle objective

The positive PSD constraint is automatic because the spline represents
`log(PSD)`. The smoothing strength `PENALTY` controls curvature, not an
arbitrary moving-window width."""),
    code("""PENALTY = 80.0

def pspline_objective(coefficients, penalty=PENALTY):
    log_psd = basis @ coefficients
    roughness = second_difference @ coefficients
    return np.sum(log_psd + periodogram * np.exp(-log_psd)) + 0.5 * penalty * np.sum(roughness**2)

initial = np.linalg.lstsq(basis, np.log(np.maximum(periodogram, np.median(periodogram) * 1e-6)), rcond=None)[0]
fit = minimize(pspline_objective, initial, method="L-BFGS-B")
if not fit.success:
    raise RuntimeError(fit.message)

log_psd_map = basis @ fit.x
psd_map = np.exp(log_psd_map)
curvature = periodogram * np.exp(-log_psd_map)
hessian = basis.T @ (curvature[:, None] * basis) + PENALTY * second_difference.T @ second_difference
coefficient_covariance = np.linalg.pinv(hessian)
draws = rng.multivariate_normal(fit.x, coefficient_covariance, size=400)
psd_draws = np.exp(draws @ basis.T)
low, high = np.quantile(psd_draws, [0.05, 0.95], axis=0)

fig, ax = plt.subplots(figsize=(9, 4))
ax.loglog(frequency, periodogram, color="0.82", lw=0.5, label="periodogram")
ax.loglog(frequency, true_psd, "k--", lw=2, label="injected PSD")
ax.loglog(frequency, psd_map, color="C3", lw=2, label="P-spline MAP")
ax.fill_between(frequency, low, high, color="C3", alpha=0.22, label="local 90% band")
ax.set(xlabel="frequency [Hz]", ylabel="PSD", title="Penalised Whittle P-spline fit")
ax.legend()
plt.show()""", figure="lisa-pspline-fit"),
    md("""## 3. Check the implied whitening

If the PSD model is adequate, `periodogram / psd_map` should fluctuate around
one without broad frequency-dependent structure. This is a numerical model
check, not proof that the physical noise components are identifiable."""),
    code("""whitened_power = periodogram / psd_map
log_bins = np.array_split(np.arange(n_frequency), 12)
bin_frequency = np.array([np.exp(np.mean(log_frequency[index])) for index in log_bins])
bin_power = np.array([np.mean(whitened_power[index]) for index in log_bins])

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.semilogx(frequency, whitened_power, ".", ms=2, alpha=0.25, label="bins")
ax.semilogx(bin_frequency, bin_power, "o-", lw=2, label="log-band mean")
ax.axhline(1, color="k", ls="--")
ax.set(xlabel="frequency [Hz]", ylabel="whitened power", title="PSD posterior-predictive diagnostic")
ax.legend()
plt.show()
print("mean whitened power:", whitened_power.mean())""", figure="lisa-pspline-whitening"),
    *_exercise(
        "Refit with penalties 2, 80, and 5000. Which fit follows periodogram "
        "spikes, and which erases the broad bump near 3 mHz?",
        """# Your code here
penalties = [2.0, 80.0, 5000.0]
fits_by_penalty = {}
raise NotImplementedError("minimise pspline_objective for each penalty")""",
        "Pass the penalty explicitly and warm-start each fit from `fit.x`. Plot "
        "the resulting PSDs against the injected PSD.",
        """for penalty in penalties:
    result = minimize(lambda beta: pspline_objective(beta, penalty), fit.x, method="L-BFGS-B")
    fits_by_penalty[penalty] = np.exp(basis @ result.x)
for penalty, estimate in fits_by_penalty.items():
    plt.loglog(frequency, estimate, label=f"lambda={penalty:g}")
plt.loglog(frequency, true_psd, "k--", label="truth")
plt.legend(); plt.show()""",
    ),
    md("""## Instrument noise and foregrounds

One observed total spectrum does not identify two unrestricted positive smooth
surfaces. Separating an instrumental PSD from Galactic confusion requires
additional structure: response-informed shapes, multiple TDI channels,
time dependence, informative regularisation, resolved-source information, or
other data. A visually good total-PSD fit alone is not component recovery."""),
]
write(
    "07_lisa_pspline_psd.ipynb",
    "Part 3C: LISA PSD estimation with P-splines",
    lisa_c_cells,
)


# Part 3D: repeat the global-fit wheel in WDM coefficients, now with a gap.
wdm_global_fit_cells = [
    md(
        r"""## 2. The same global fit, now in the WDM domain

Nothing about blocked Gibbs sampling belongs specifically to a Fourier basis.
Module 06 used frequency-domain residuals. Here the data model is unchanged,

$$
d_{nm}=\sum_{i=1}^{3} a_i h^{(i)}_{nm}+n_{nm},
$$

but every datum, template, and residual is a WDM coefficient. The conditional
for amplitude $a_i$ uses the shared residual

$$
r^{(i)}_{nm}=d_{nm}-\sum_{j\ne i}a_jh^{(j)}_{nm},
$$

and the same Gaussian update as Module 06, with the **masked WDM inner
product** replacing the frequency-domain inner product. Columns in the gap and
a conservative edge guard have infinite uncertainty operationally: they are
excluded from every source and noise update.

This is a controlled counterpart to Module 06's fixed-shape example. The
source shapes and source count are known; only their amplitudes and a common
noise scale are inferred. The point is to expose residual handoff with missing
data, not to claim a production LISA global fit."""
    ),
    md(
        """**Predict before running:** Does localising the gap in a handful of WDM
columns make the three overlapping sources independent of one another? What
information should the gap remove from every conditional update?"""
    ),
    code(
        r'''# Three nearly coincident binaries: visually they share one WDM ridge, but
# their phase evolution still lets the joint likelihood distinguish them.
WDM_GLOBAL_F0 = 3.0e-3 + np.array([0.0, 0.25e-6, 0.50e-6])
WDM_GLOBAL_PHASE = np.array([0.2, 0.5, 0.8])
WDM_GLOBAL_TRUTH = np.array([1.0, 0.75, 0.55])
WDM_GLOBAL_BASE_SNR = 30.0

wdm_global_time_templates = np.asarray([
    np.sin(2 * np.pi * frequency * toy_time + phase0)
    for frequency, phase0 in zip(WDM_GLOBAL_F0, WDM_GLOBAL_PHASE)
])
wdm_global_templates = np.asarray([
    np.asarray(
        WDM.from_time_series(
            WDMTimeSeries(time_template, dt=cadence), nt=WDM_NT
        ).coeffs[0]
    )
    for time_template in wdm_global_time_templates
])

# A hard gap rings beyond the literally missing samples. The introductory lab
# deliberately shows that edge leakage; inference uses a wider, visible guard
# so those contaminated columns do not enter any source or noise block.
WDM_GLOBAL_GAP_GUARD_DAYS = 0.75
wdm_global_pixel_time = np.arange(WDM_NT) * wdm_data.delta_t / 86400
wdm_global_gap_pixels = (
    (wdm_global_pixel_time >= GAP_DAYS[0] - WDM_GLOBAL_GAP_GUARD_DAYS)
    & (wdm_global_pixel_time <= GAP_DAYS[1] + WDM_GLOBAL_GAP_GUARD_DAYS)
)
wdm_global_masked_var = wdm_stationary_var.copy()
wdm_global_masked_var[wdm_global_gap_pixels, :] = np.nan


def wdm_global_inner(first, second, variance=wdm_global_masked_var):
    """Gap-aware diagonal WDM inner product used by every block."""
    return wdm_inner_product(first, second, variance)


# Give each unit-amplitude template the same retained-data SNR. Scaling the time
# series and its WDM coefficients together preserves their exact relationship.
template_normalisations = WDM_GLOBAL_BASE_SNR / np.sqrt([
    wdm_global_inner(template, template) for template in wdm_global_templates
])
wdm_global_time_templates *= template_normalisations[:, None]
wdm_global_templates *= template_normalisations[:, None, None]

wdm_global_rng = np.random.default_rng(20260831)
wdm_global_noise_time = wdm_global_rng.normal(size=toy_time.size)
wdm_global_data_time = (
    np.sum(WDM_GLOBAL_TRUTH[:, None] * wdm_global_time_templates, axis=0)
    + wdm_global_noise_time
)
wdm_global_data_time[~available] = 0.0
wdm_global_data = np.asarray(
    WDM.from_time_series(
        WDMTimeSeries(wdm_global_data_time, dt=cadence), nt=WDM_NT
    ).coeffs[0]
)

# wdm_inner_product excludes the DC and Nyquist edge channels. Count the same
# real Gaussian coefficients for the exact inverse-gamma noise update below.
wdm_global_n_valid = np.isfinite(wdm_global_masked_var[:, 1:-1]).sum()

wdm_global_precision = np.asarray([
    [wdm_global_inner(first, second) for second in wdm_global_templates]
    for first in wdm_global_templates
])
wdm_global_rhs = np.asarray([
    wdm_global_inner(template, wdm_global_data)
    for template in wdm_global_templates
])
wdm_global_joint = np.linalg.solve(wdm_global_precision, wdm_global_rhs)
wdm_global_covariance = np.linalg.inv(wdm_global_precision)

# The same templates without a gap quantify the information actually lost.
wdm_global_full_precision = np.asarray([
    [wdm_inner_product(first, second, wdm_stationary_var)
     for second in wdm_global_templates]
    for first in wdm_global_templates
])
wdm_global_full_covariance = np.linalg.inv(wdm_global_full_precision)
uncertainty_inflation = np.sqrt(
    np.diag(wdm_global_covariance) / np.diag(wdm_global_full_covariance)
)

print("injected amplitudes        ", WDM_GLOBAL_TRUTH)
print("masked joint estimate      ", np.round(wdm_global_joint, 3))
print("masked posterior sigma     ", np.round(np.sqrt(np.diag(wdm_global_covariance)), 3))
print("gap/full sigma ratio       ", np.round(uncertainty_inflation, 3))
print(f"masked WDM columns          {wdm_global_gap_pixels.sum()}/{WDM_NT} "
      f"({wdm_global_gap_pixels.mean():.1%}, including the edge guard)")'''
    ),
    md(
        r"""### One pass versus a blocked chain

A WDM representation localises the missing interval; it does **not** make
overlapping source templates orthogonal. A one-pass subtraction still hands
every fitting error to the next block and never revisits it.

The blocked chain below cycles through the three amplitudes and then the noise
scale. With a flat amplitude prior, each source conditional is Gaussian. If
$\eta$ multiplies the WDM variance, the final block is also an exact draw:

$$
\eta\mid r \sim \mathrm{InvGamma}\!\left(
\frac{N_{\rm valid}}{2},\frac{(r\mid r)_{\eta=1}}{2}\right).
$$

Only valid, non-edge WDM pixels enter $N_{\rm valid}$ or the residual norm."""
    ),
    code(
        r'''# One forward pass, retained as the deliberately order-dependent baseline.
wdm_one_pass = np.zeros(3)
wdm_one_pass_residual = wdm_global_data.copy()
for source_index, template in enumerate(wdm_global_templates):
    wdm_one_pass[source_index] = (
        wdm_global_inner(template, wdm_one_pass_residual)
        / wdm_global_inner(template, template)
    )
    wdm_one_pass_residual -= wdm_one_pass[source_index] * template

# Exact blocked Gibbs updates for amplitudes plus an exact noise-scale block.
WDM_GIBBS_SWEEPS = 1600
WDM_GIBBS_BURN_IN = 300
wdm_gibbs_state = np.zeros(3)
wdm_gibbs_noise_scale = 1.0
wdm_gibbs_history = [wdm_gibbs_state.copy()]
wdm_gibbs_noise_history = []
wdm_gibbs_substates = []
wdm_gibbs_active_blocks = []
wdm_gibbs_conditional_means = []
wdm_gibbs_conditional_sds = []

for sweep in range(WDM_GIBBS_SWEEPS):
    for source_index, template in enumerate(wdm_global_templates):
        conditional_residual = (
            wdm_global_data
            - np.sum(
                wdm_gibbs_state[:, None, None] * wdm_global_templates, axis=0
            )
            + wdm_gibbs_state[source_index] * template
        )
        base_precision = wdm_global_inner(template, template)
        conditional_mean = (
            wdm_global_inner(template, conditional_residual) / base_precision
        )
        conditional_sd = np.sqrt(wdm_gibbs_noise_scale / base_precision)
        wdm_gibbs_state[source_index] = wdm_global_rng.normal(
            conditional_mean, conditional_sd
        )
        wdm_gibbs_substates.append(wdm_gibbs_state.copy())
        wdm_gibbs_active_blocks.append(source_index)
        wdm_gibbs_conditional_means.append(conditional_mean)
        wdm_gibbs_conditional_sds.append(conditional_sd)

    current_residual = wdm_global_data - np.sum(
        wdm_gibbs_state[:, None, None] * wdm_global_templates, axis=0
    )
    residual_power = wdm_global_inner(current_residual, current_residual)
    wdm_gibbs_noise_scale = (residual_power / 2) / wdm_global_rng.gamma(
        wdm_global_n_valid / 2, 1.0
    )
    wdm_gibbs_history.append(wdm_gibbs_state.copy())
    wdm_gibbs_noise_history.append(wdm_gibbs_noise_scale)

wdm_gibbs_history = np.asarray(wdm_gibbs_history)
wdm_gibbs_noise_history = np.asarray(wdm_gibbs_noise_history)
wdm_gibbs_substates = np.asarray(wdm_gibbs_substates)
wdm_gibbs_active_blocks = np.asarray(wdm_gibbs_active_blocks)
wdm_gibbs_conditional_means = np.asarray(wdm_gibbs_conditional_means)
wdm_gibbs_conditional_sds = np.asarray(wdm_gibbs_conditional_sds)
wdm_gibbs_samples = wdm_gibbs_history[WDM_GIBBS_BURN_IN:]
wdm_gibbs_noise_samples = wdm_gibbs_noise_history[WDM_GIBBS_BURN_IN:]

wdm_final_residual = wdm_global_data - np.sum(
    wdm_gibbs_samples.mean(axis=0)[:, None, None] * wdm_global_templates, axis=0
)

print("true             ", np.round(WDM_GLOBAL_TRUTH, 3))
print("one pass         ", np.round(wdm_one_pass, 3))
print("joint mean       ", np.round(wdm_global_joint, 3))
print("Gibbs mean       ", np.round(wdm_gibbs_samples.mean(axis=0), 3))
print("Gibbs uncertainty", np.round(wdm_gibbs_samples.std(axis=0), 3))
print(f"noise scale       {wdm_gibbs_noise_samples.mean():.3f} "
      f"+/- {wdm_gibbs_noise_samples.std():.3f}")
print(f"final (r|r)/N_valid = "
      f"{wdm_global_inner(wdm_final_residual, wdm_final_residual)/wdm_global_n_valid:.3f}")'''
    ),
    code(
        r'''fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), layout="constrained")

for source_index in range(3):
    axes[0, 0].plot(
        wdm_gibbs_history[:100, source_index],
        lw=0.9,
        color=f"C{source_index}",
        label=f"source {source_index + 1}",
    )
    axes[0, 0].axhline(
        WDM_GLOBAL_TRUTH[source_index],
        color=f"C{source_index}",
        ls="--",
        alpha=0.6,
    )
    axes[0, 1].hist(
        wdm_gibbs_samples[:, source_index],
        bins=34,
        density=True,
        histtype="step",
        color=f"C{source_index}",
        label=f"source {source_index + 1}",
    )
    axes[0, 1].axvline(
        wdm_global_joint[source_index],
        color=f"C{source_index}",
        ls="--",
        alpha=0.7,
    )
axes[0, 0].set(
    xlabel="Gibbs sweep", ylabel="amplitude multiplier", title="WDM conditional chains"
)
axes[0, 1].set(
    xlabel="amplitude multiplier", ylabel="posterior density", title="Masked WDM posterior"
)
axes[0, 0].legend(fontsize=8)
axes[0, 1].legend(fontsize=8)

wdm_global_show = (wdm_frequency > 2.75e-3) & (wdm_frequency < 3.25e-3)
data_display = np.abs(wdm_global_data[:, wdm_global_show].T)
residual_display = np.abs(wdm_final_residual[:, wdm_global_show].T)
data_display[:, wdm_global_gap_pixels] = np.nan
residual_display[:, wdm_global_gap_pixels] = np.nan
global_colour_scale = np.nanpercentile(data_display, 99.5)

for ax, image_data, title in zip(
    axes[1],
    (data_display, residual_display),
    ("shared WDM data", "posterior-mean residual"),
):
    image = ax.pcolormesh(
        wdm_global_pixel_time,
        1e3 * wdm_frequency[wdm_global_show],
        image_data,
        shading="nearest",
        cmap="magma",
        vmin=0,
        vmax=global_colour_scale,
    )
    mark_gap(
        ax,
        GAP_DAYS[0] - WDM_GLOBAL_GAP_GUARD_DAYS,
        GAP_DAYS[1] + WDM_GLOBAL_GAP_GUARD_DAYS,
        label="gap + edge guard",
    )
    ax.set(xlabel="mission time [days]", ylabel="frequency [mHz]", title=title)
axes[1, 0].legend(fontsize=8, loc="upper right")
fig.colorbar(image, ax=axes[1], label=r"$|w_{nm}|$", pad=0.02)
plt.show()''',
        figure="lisa-wdm-global-fit",
    ),
    md(
        r"""### Animation: the WDM residual handed from block to block

As in Module 06, every frame is one conditional source update rather than one
completed sweep. The parameter-space move is on the left. On the right, the
same shared WDM residual is updated and passed to the next source block. The
hatched gap-plus-guard interval is absent from **every** update; Gibbs sampling
does not invent coefficients inside the missing interval."""
    ),
    code(
        r'''wdm_animation_start = 12
wdm_animation_stop = wdm_animation_start + 30
wdm_animation_states = wdm_gibbs_substates[
    wdm_animation_start:wdm_animation_stop
]
wdm_animation_blocks = wdm_gibbs_active_blocks[
    wdm_animation_start:wdm_animation_stop
]
wdm_animation_means = wdm_gibbs_conditional_means[
    wdm_animation_start:wdm_animation_stop
]
wdm_animation_sds = wdm_gibbs_conditional_sds[
    wdm_animation_start:wdm_animation_stop
]
wdm_animation_initial = wdm_gibbs_substates[wdm_animation_start - 1]
wdm_animation_path = np.vstack([wdm_animation_initial, wdm_animation_states])

wdm_covariance_12 = wdm_global_covariance[:2, :2]
wdm_sigma_12 = np.sqrt(np.diag(wdm_covariance_12))
wdm_a1_grid = np.linspace(
    wdm_global_joint[0] - 4 * wdm_sigma_12[0],
    wdm_global_joint[0] + 4 * wdm_sigma_12[0],
    120,
)
wdm_a2_grid = np.linspace(
    wdm_global_joint[1] - 4 * wdm_sigma_12[1],
    wdm_global_joint[1] + 4 * wdm_sigma_12[1],
    120,
)
WDM_A1_GRID, WDM_A2_GRID = np.meshgrid(wdm_a1_grid, wdm_a2_grid)
wdm_offsets_12 = np.stack(
    [WDM_A1_GRID - wdm_global_joint[0], WDM_A2_GRID - wdm_global_joint[1]],
    axis=-1,
)
wdm_mahalanobis_12 = np.einsum(
    "...i,ij,...j->...",
    wdm_offsets_12,
    np.linalg.inv(wdm_covariance_12),
    wdm_offsets_12,
)

fig, (wdm_posterior_ax, wdm_residual_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
wdm_posterior_ax.contour(
    WDM_A1_GRID,
    WDM_A2_GRID,
    np.exp(-0.5 * wdm_mahalanobis_12),
    levels=np.exp(-0.5 * np.array([9.0, 4.0, 1.0])),
    colors=".65",
)
wdm_posterior_ax.plot(
    WDM_GLOBAL_TRUTH[0], WDM_GLOBAL_TRUTH[1], "*", color="k", ms=10, label="injected"
)
wdm_posterior_ax.plot(
    wdm_global_joint[0], wdm_global_joint[1], "+", color="C3", ms=10,
    mew=2, label="joint mean"
)
(wdm_path_line,) = wdm_posterior_ax.plot([], [], color=".7", lw=1)
(wdm_step_line,) = wdm_posterior_ax.plot([], [], lw=3)
(wdm_point,) = wdm_posterior_ax.plot([], [], "o", color="k", ms=5)
wdm_posterior_ax.set(
    xlabel="source 1 amplitude", ylabel="source 2 amplitude",
    title="conditional moves through the WDM posterior"
)
wdm_posterior_ax.legend(fontsize=8)

wdm_initial_animation_residual = wdm_global_data.copy()
wdm_initial_animation_display = np.abs(
    wdm_initial_animation_residual[:, wdm_global_show].T
)
wdm_initial_animation_display[:, wdm_global_gap_pixels] = np.nan
wdm_residual_image = wdm_residual_ax.pcolormesh(
    wdm_global_pixel_time,
    1e3 * wdm_frequency[wdm_global_show],
    wdm_initial_animation_display,
    shading="nearest",
    cmap="magma",
    vmin=0,
    vmax=global_colour_scale,
)
mark_gap(
    wdm_residual_ax,
    GAP_DAYS[0] - WDM_GLOBAL_GAP_GUARD_DAYS,
    GAP_DAYS[1] + WDM_GLOBAL_GAP_GUARD_DAYS,
    label=None,
)
wdm_conditional_text = wdm_residual_ax.text(
    0.02, 0.95, "", transform=wdm_residual_ax.transAxes, va="top",
    bbox=dict(facecolor="white", alpha=0.88, edgecolor="none")
)
wdm_residual_ax.set(
    xlabel="mission time [days]", ylabel="frequency [mHz]",
    title="shared WDM residual"
)


def animate_wdm_gibbs_block(frame):
    state = wdm_animation_states[frame]
    previous = wdm_animation_path[frame]
    active = int(wdm_animation_blocks[frame])
    colour = f"C{active}"
    wdm_path_line.set_data(
        wdm_animation_path[: frame + 2, 0], wdm_animation_path[: frame + 2, 1]
    )
    wdm_step_line.set_data([previous[0], state[0]], [previous[1], state[1]])
    wdm_step_line.set_color(colour)
    wdm_point.set_data([state[0]], [state[1]])

    residual_after = wdm_global_data - np.sum(
        state[:, None, None] * wdm_global_templates, axis=0
    )
    residual_display = np.abs(residual_after[:, wdm_global_show].T)
    residual_display[:, wdm_global_gap_pixels] = np.nan
    wdm_residual_image.set_array(residual_display.ravel())
    wdm_conditional_text.set_text(
        f"draw a{active + 1} from its conditional\n"
        f"mean {wdm_animation_means[frame]:.3f}, "
        f"sd {wdm_animation_sds[frame]:.3f}\n"
        f"new value {state[active]:.3f}"
    )
    wdm_posterior_ax.set_title(f"block {active + 1}: condition on the other sources")
    wdm_residual_ax.set_title(f"pass this residual to block {(active + 1) % 3 + 1}")
    return wdm_path_line, wdm_step_line, wdm_point, wdm_residual_image, wdm_conditional_text


wdm_gibbs_animation = FuncAnimation(
    fig,
    animate_wdm_gibbs_block,
    frames=len(wdm_animation_states),
    interval=320,
)
plt.close(fig)
show_animation(wdm_gibbs_animation)'''
    ),
    md(
        r""":::{admonition} What WDM changes—and what it does not
:class: warning

- The gap is localised, so most WDM pixels remain usable rather than every
  Fourier bin being contaminated by a mission-long window.
- The likelihood, source blocks, and noise block all use exactly the same mask.
- Source coupling remains: one-pass subtraction is still order-dependent, and
  an imperfect source model still leaves structure for later blocks to absorb.
- Masking whole columns discards more than the literal missing samples. Gap-edge
  pixels can remain correlated. The 0.75-day guard is a visible conservative
  teaching choice, not a calibrated production rule, so this diagonal
  likelihood is not a complete gap-marginalised analysis.
:::"""
    ),
]


# Part 3D: the validated WDM laboratory is now independently runnable.
lisa_d_cells = [
    _module_intro(
        "Repeat the global-fit Gibbs wheel in a WDM time-frequency representation "
        "with a real gap mask and a shared residual.",
        "Run the gap/WDM laboratory, then compare one-pass subtraction with the "
        "masked WDM Gibbs chain.",
        "The diagonal WDM likelihood is exact only under its covariance "
        "assumptions. Gaps and non-stationarity can correlate pixels.",
    ),
    *lisa_setup,
    *_renumber(_between(lisa_source, "## 3. Real LISA data will be more complicated", "## 4. Inner product, SNR, and likelihood"), {"3": "1", "4": "2"}),
    *wdm_global_fit_cells,
    *_exercise(
        "Reverse the one-pass source order and compare its masked residual norm "
        "with the forward pass and the joint fit. Why does WDM gap localisation "
        "not remove order dependence between overlapping sources?",
        """reverse_order = [2, 1, 0]
# Your code here: repeat the one-pass updates in reverse_order.
print("Exercise ready:", reverse_order)""",
        "Each source is fitted to a residual containing the current errors from "
        "the other sources. WDM localises the gap, not source-model error.",
        """reverse_residual = wdm_global_data.copy()
reverse_amplitudes = np.zeros(3)
for source_index in reverse_order:
    template = wdm_global_templates[source_index]
    reverse_amplitudes[source_index] = (
        wdm_global_inner(template, reverse_residual)
        / wdm_global_inner(template, template)
    )
    reverse_residual -= reverse_amplitudes[source_index] * template

joint_residual = wdm_global_data - np.sum(
    wdm_global_joint[:, None, None] * wdm_global_templates, axis=0
)
print("forward one-pass norm:", wdm_global_inner(wdm_one_pass_residual, wdm_one_pass_residual))
print("reverse one-pass norm:", wdm_global_inner(reverse_residual, reverse_residual))
print("joint-fit norm:", wdm_global_inner(joint_residual, joint_residual))""",
    ),
]
write(
    "08_lisa_wdm_time_frequency.ipynb",
    "Part 3D: LISA WDM time-frequency analysis",
    lisa_d_cells,
)


# Supplement: a genuinely blind student worksheet and a separate worked answer.
# The answer is built by a hidden toctree in index.md, but is deliberately not
# listed in _toc.yml.  The HDF5 strain is generated reproducibly by
# scripts/build_lvk_blind_challenge_data.py and contains no injection metadata.
LVK_CHALLENGE_DATA_SETUP = [
    code(
        """import os, sys, subprocess, importlib.util

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
missing = [
    package
    for package in ("corner", "h5py")
    if importlib.util.find_spec(package) is None
]
if missing:
    if IN_COLAB:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "corner>=2.2",
            "h5py>=3.11",
        ])
    else:
        raise ImportError(
            "Install corner>=2.2 and h5py>=3.11, "
            "or use the locked workshop environment."
        )"""
    ),
    code(
        """from pathlib import Path
from urllib.request import urlretrieve

import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, spectrogram, welch

plt.style.use("seaborn-v0_8-whitegrid")

local_candidates = [
    Path("assets/lvk_blind_challenge.h5"),
    Path("../assets/lvk_blind_challenge.h5"),
]
DATA_PATH = next((path for path in local_candidates if path.exists()), local_candidates[0])
DATA_URL = (
    "https://raw.githubusercontent.com/nz-gravity/"
    "FQCP2026_GW_data_analysis/main/assets/lvk_blind_challenge.h5"
)
if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(DATA_URL, DATA_PATH)

with h5py.File(DATA_PATH, "r") as source:
    sampling_frequency = int(source.attrs["sampling_frequency_hz"])
    duration = float(source.attrs["duration_s"])
    start_time = float(source.attrs["start_time_s"])
    strain = {
        detector: np.asarray(source[f"strain/{detector}"][:], dtype=np.float64)
        for detector in ("H1", "L1")
    }

time = start_time + np.arange(len(strain["H1"])) / sampling_frequency
assert len(strain["H1"]) == len(strain["L1"]) == int(duration * sampling_frequency)
assert all(np.isfinite(values).all() for values in strain.values())
print(f"Loaded {duration:.0f} s at {sampling_frequency} Hz from {DATA_PATH}")
print("Detector arrays:", {name: values.shape for name, values in strain.items()})"""
    ),
]


def challenge_task(number, title, prompt, starter):
    """A runnable blind task with no answer embedded in the student notebook."""
    return [
        md(f"""## {number}. {title}

{prompt}"""),
        code(starter),
    ]


student_challenge_cells = [
    _module_intro(
        "Search two synthetic detector streams for compact-binary signals, "
        "separate a detector transient from an astrophysical coincidence, "
        "estimate off-source PSDs, and run restricted parameter estimation.",
        "Complete Tasks 1--7 in order. Record candidate times before opening "
        "any instructor material.",
        "This is a controlled classroom challenge: Gaussian line-free noise, "
        "Newtonian inspiral teaching signals, fixed response and mass ratio for PE, no "
        "calibration uncertainty, and no false-alarm-rate claim.",
    ),
    md(
        r"""## Rules and supplied search ranges

The HDF5 file contains only H1/L1 strain and acquisition metadata. It may
contain more than one CBC and at least one data-quality problem. The events do
not overlap.

Use two template-bank ranges:

- bank A: $\mathcal M\in[16.5,26.0]M_\odot$;
- bank B: $\mathcal M\in[26.6,41.0]M_\odot$.

Use $q=0.9$ for the inexpensive search bank. For the two-dimensional PE, the
instructor supplies a fixed mass ratio for each candidate. These search ranges
are not detection-rate priors, and the ranking statistic below is not assigned
a false-alarm rate.

### What you do—and what is supplied

You will **choose and justify analysis settings, run the search, read trigger
tables, diagnose the transient, define the two priors, and interpret posterior
and residual plots**. You are not expected to derive or debug an FFT matched
filter or MCMC implementation on your first day. The waveform, bank runner,
coincidence bookkeeping, likelihood, sampler, and plotting scaffolds are
provided and deliberately kept visible so you can connect the equations to the
code.

[Download the HDF5 data directly](../assets/lvk_blind_challenge.h5)."""
    ),
    *[deepcopy(cell) for cell in LVK_CHALLENGE_DATA_SETUP],
    *challenge_task(
        1,
        "Inspect the data",
        "Check shapes, finite values, and detector-by-detector scale. Plot a "
        "downsampled overview, but do not expect a CBC to be visible in raw strain. "
        "**Tip:** failure to see a chirp here is expected, not evidence that the "
        "file is empty. **Suggested plot:** two aligned time-series panels with "
        "the same time axis.",
        """# Your overview here. Keep the plot light by downsampling.
step = sampling_frequency // 8
fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
for axis, detector in zip(axes, ("H1", "L1")):
    axis.plot(time[::step], strain[detector][::step], lw=0.35)
    axis.set(ylabel=f"{detector} strain")
axes[-1].set_xlabel("time after file start [s]")
plt.show()""",
    ),
    *challenge_task(
        2,
        "Choose the analysis duration",
        r"Use the lowest chirp mass in each bank and $f_\mathrm{low}=20$ Hz. "
        "Compute the leading-order inspiral time and justify an FFT-friendly "
        "analysis duration no longer than 8 seconds. **Tip:** round upward to a "
        "convenient power-of-two duration and leave room around coalescence. "
        "**Optional plot:** inspiral duration versus chirp mass across both banks.",
        r"""MTSUN_SI = 4.925490947e-6


def inspiral_time(chirp_mass, f_low=20.0):
    return (
        5
        / 256
        * (MTSUN_SI * chirp_mass) ** (-5 / 3)
        * (np.pi * f_low) ** (-8 / 3)
    )


for lower_edge in (16.5, 26.6):
    print(lower_edge, inspiral_time(lower_edge), "s")

analysis_duration = 8.0  # supplied default; explain why it is adequate
print("Chosen analysis duration:", analysis_duration)

mass_grid = np.linspace(16.5, 41.0, 200)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(mass_grid, inspiral_time(mass_grid))
ax.axhline(analysis_duration, color="C3", ls="--", label="chosen segment")
ax.set(xlabel=r"chirp mass [$M_\\odot$]", ylabel="time from 20 Hz [s]")
ax.legend()
plt.show()""",
    ),
    md(
        r"""### How the supplied template bank works

Each template is a deliberately simplified Newtonian inspiral,

$$
\tilde h(f;\mathcal M,t_c)\propto f^{-7/6}
\exp\left[i\left(-\frac{\pi}{4}
+\frac{3}{128}(\pi\mathcal M_{\rm sec}f)^{-5/3}
-2\pi f t_c\right)\right],
$$

with a smooth cutoff near the mass-dependent ISCO. The supplied bank runner:

1. chooses a grid of chirp masses;
2. builds one waveform for each grid point;
3. correlates it with each detector using the PSD-weighted matched filter;
4. keeps the largest SNR and corresponding template mass at every time.

This is a teaching bank, not a production LVK search. On the first pass, focus
on what goes into `run_template_bank` and what it returns; the FFT details are
optional reading."""
    ),
    *challenge_task(
        3,
        "Build an initial PSD and run the supplied search",
        "Matched filtering needs a PSD before the event times are known. Start "
        "with a robust median-Welch estimate from the full record. The toy "
        "waveform, matched filter, and template-bank loop are supplied below: "
        "read their inputs and outputs, then run them. Your job is to identify "
        "the common-detector peaks and explain why maximising over the bank is "
        "useful. **Suggested plot:** bank-maximum H1/L1 SNR versus time for each "
        "mass range, with an exploratory threshold at 6.",
        """WELCH_SECONDS = 4


def median_welch(values):
    return welch(
        values,
        fs=sampling_frequency,
        window=("tukey", 0.2),
        nperseg=WELCH_SECONDS * sampling_frequency,
        noverlap=WELCH_SECONDS * sampling_frequency // 2,
        detrend=False,
        average="median",
    )


initial_psd = {detector: median_welch(values) for detector, values in strain.items()}

# --- Supplied first-day search machinery ---
n_samples = len(time)
frequency = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)
frequency_spacing = 1 / duration
data_frequency = {
    detector: np.fft.rfft(values) / sampling_frequency
    for detector, values in strain.items()
}
psd_on_search_grid = {
    detector: np.interp(frequency, *initial_psd[detector])
    for detector in ("H1", "L1")
}


def component_masses(chirp_mass, mass_ratio):
    eta = mass_ratio / (1 + mass_ratio) ** 2
    total_mass = chirp_mass / eta ** (3 / 5)
    primary_mass = total_mass / (1 + mass_ratio)
    return primary_mass, mass_ratio * primary_mass


def newtonian_chirp(frequency_array, chirp_mass, mass_ratio, coalescence_time=0.0):
    '''Supplied toy frequency-domain inspiral with a smooth ISCO cutoff.'''
    waveform = np.zeros(frequency_array.size, dtype=complex)
    primary_mass, secondary_mass = component_masses(chirp_mass, mass_ratio)
    f_isco = 1 / (6 ** 1.5 * np.pi * MTSUN_SI * (primary_mass + secondary_mass))
    taper_start = 0.85 * f_isco
    usable = (frequency_array >= 20.0) & (frequency_array < f_isco)
    taper = np.ones(frequency_array.size)
    taper_region = (frequency_array >= taper_start) & (frequency_array < f_isco)
    taper[taper_region] = 0.5 * (
        1 + np.cos(np.pi * (frequency_array[taper_region] - taper_start)
                   / (f_isco - taper_start))
    )
    phase = (
        -np.pi / 4
        + 3 / 128 * (np.pi * MTSUN_SI * chirp_mass
                     * frequency_array[usable]) ** (-5 / 3)
        - 2 * np.pi * frequency_array[usable] * coalescence_time
    )
    waveform[usable] = frequency_array[usable] ** (-7 / 6) * taper[usable] * np.exp(1j * phase)
    return waveform


def matched_filter_snr(data_fd, template_fd, psd):
    '''Return phase-maximised SNR as a function of coalescence time.'''
    usable = (frequency >= 20) & (frequency <= 400) & np.isfinite(psd) & (psd > 0)
    integrand = np.zeros(frequency.size, dtype=complex)
    integrand[usable] = data_fd[usable] * np.conj(template_fd[usable]) / psd[usable]
    padded = np.zeros(n_samples, dtype=complex)
    padded[: frequency.size] = integrand
    correlation = 4 * frequency_spacing * n_samples * np.fft.ifft(padded)
    norm = np.sqrt(4 * frequency_spacing * np.sum(
        np.abs(template_fd[usable]) ** 2 / psd[usable]
    ))
    return np.abs(correlation) / norm


def run_template_bank(chirp_masses, mass_ratio=0.9):
    '''Maximise the SNR time series over a supplied chirp-mass grid.'''
    maximum_snr = {detector: np.zeros(n_samples) for detector in ("H1", "L1")}
    maximum_mass = {detector: np.zeros(n_samples) for detector in ("H1", "L1")}
    for chirp_mass in chirp_masses:
        template = newtonian_chirp(frequency, chirp_mass, mass_ratio)
        for detector in ("H1", "L1"):
            trial = matched_filter_snr(
                data_frequency[detector], template, psd_on_search_grid[detector]
            )
            better = trial > maximum_snr[detector]
            maximum_snr[detector][better] = trial[better]
            maximum_mass[detector][better] = chirp_mass
    return maximum_snr, maximum_mass


banks = {
    "A: 16.5--26": np.linspace(16.5, 26.0, 20),
    "B: 26.6--41": np.linspace(26.6, 41.0, 24),
}
search_snr, best_mass = {}, {}
for bank_name, chirp_masses in banks.items():
    search_snr[bank_name], best_mass[bank_name] = run_template_bank(chirp_masses)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for axis, bank_name in zip(axes, banks):
    for detector in ("H1", "L1"):
        axis.plot(time, search_snr[bank_name][detector], lw=0.8, label=detector)
    axis.axhline(6, color="k", ls="--", lw=0.8)
    axis.set(ylabel="bank-max SNR", title=bank_name, xlim=(145, 235))
    axis.legend()
axes[-1].set_xlabel("time after file start [s]")
plt.show()""",
    ),
    *challenge_task(
        4,
        "Apply coincidence and handle the transient",
        "List single-detector peaks above your exploratory threshold. Pair H1 "
        "and L1 triggers within 20 ms. Inspect the unmatched loud feature in a "
        "time-frequency plot and state whether you will veto, gate, inpaint, or "
        "model it. Do not use it in a PSD estimate. The peak finder and "
        "coincidence bookkeeping are supplied; focus on reading the table. "
        "**Suggested plots:** the SNR-time plot above and matched H1/L1 "
        "spectrograms spanning ten seconds around the unmatched feature.",
        """COINCIDENCE_WINDOW = 0.020
SNR_THRESHOLD = 6.0
peak_tables = {}
coincident_candidates = []
for bank_name in banks:
    peak_tables[bank_name] = {}
    for detector in ("H1", "L1"):
        peaks, _ = find_peaks(
            search_snr[bank_name][detector],
            height=SNR_THRESHOLD,
            distance=int(0.5 * sampling_frequency),
        )
        peak_tables[bank_name][detector] = peaks
        print(f"\\n{bank_name} {detector} peaks")
        for peak in peaks:
            print(
                f"  t={time[peak]:8.3f} s  "
                f"rho={search_snr[bank_name][detector][peak]:5.1f}  "
                f"Mc~{best_mass[bank_name][detector][peak]:5.1f}"
            )
    for h1_peak in peak_tables[bank_name]["H1"]:
        l1_peaks = peak_tables[bank_name]["L1"]
        separations = np.abs(time[l1_peaks] - time[h1_peak])
        if len(separations) and separations.min() <= COINCIDENCE_WINDOW:
            l1_peak = l1_peaks[np.argmin(separations)]
            candidate_time = 0.5 * (time[h1_peak] + time[l1_peak])
            coincident_candidates.append((bank_name, candidate_time, h1_peak, l1_peak))

# Cluster duplicate triggers from the two overlapping banks.
unique_candidates = []
for candidate in sorted(coincident_candidates, key=lambda item: item[1]):
    bank_name, candidate_time, h1_peak, l1_peak = candidate
    rank = np.hypot(
        search_snr[bank_name]["H1"][h1_peak],
        search_snr[bank_name]["L1"][l1_peak],
    )
    if unique_candidates and abs(candidate_time - unique_candidates[-1][1]) < 0.2:
        if rank > unique_candidates[-1][-1]:
            unique_candidates[-1] = (*candidate, rank)
    else:
        unique_candidates.append((*candidate, rank))

print("\\nTime-clustered coincidences")
for bank_name, candidate_time, h1_peak, l1_peak, rank in unique_candidates:
    print(f"  {bank_name}: t~{candidate_time:.3f} s, network rank={rank:.1f}")

# The loud single-detector feature is deliberately away from either CBC.
transient_window = (time >= 185) & (time <= 195)
sos = butter(4, (20, 250), btype="bandpass", fs=sampling_frequency, output="sos")
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
for axis, detector in zip(axes, ("H1", "L1")):
    filtered = sosfiltfilt(sos, strain[detector][transient_window])
    filtered /= np.std(filtered)
    spec_frequency, spec_time, power = spectrogram(
        filtered, fs=sampling_frequency, window=("tukey", 0.2),
        nperseg=128, noverlap=112, mode="psd",
    )
    band = (spec_frequency >= 20) & (spec_frequency <= 250)
    image = axis.pcolormesh(
        185 + spec_time, spec_frequency[band],
        10 * np.log10(power[band] + 1e-12), shading="auto",
    )
    axis.set(ylabel="frequency [Hz]", title=detector)
axes[-1].set_xlabel("time after file start [s]")
fig.colorbar(image, ax=axes, label="relative power [dB]")
plt.show()

print("Write 2--3 sentences: why is the unmatched feature not a CBC candidate,")
print("and which treatment is sufficient given its separation from the signals?")""",
    ),
    *challenge_task(
        5,
        "Estimate final off-source PSDs",
        "After locating the earliest signal, choose 128 clean seconds before it. "
        "Use four-second Tukey-windowed periodograms, 50% overlap, and median "
        "averaging. The calculation is supplied; explain why this interval is "
        "off-source and count the contributing periodograms. **Suggested plot:** "
        "H1 and L1 ASD on log-log axes from 15--450 Hz.",
        """clean = time < 128.0  # justify this only after locating the candidates
final_psd = {
    detector: median_welch(values[clean]) for detector, values in strain.items()
}
number_of_averages = 1 + (
    clean.sum() - WELCH_SECONDS * sampling_frequency
) // (WELCH_SECONDS * sampling_frequency // 2)
print("Final PSD segment length:", clean.sum() / sampling_frequency, "s")
print("Overlapping periodograms:", number_of_averages)

fig, ax = plt.subplots(figsize=(9, 4))
for detector, (frequency_welch, psd_welch) in final_psd.items():
    usable = (frequency_welch >= 15) & (frequency_welch <= 450)
    ax.loglog(frequency_welch[usable], np.sqrt(psd_welch[usable]), label=detector)
ax.set(xlabel="frequency [Hz]", ylabel=r"ASD [1/$\\sqrt{\\mathrm{Hz}}$]")
ax.legend()
plt.show()""",
    ),
    *challenge_task(
        6,
        "Plot the PE priors",
        "For each candidate, plot the chirp-mass and coalescence-time priors "
        "before sampling. The response, mass ratio, amplitude (hence distance), "
        "and phase are supplied and held fixed. The code below converts your "
        "coincidences into the two PE specifications. **Suggested plot:** four "
        "one-dimensional prior histograms in a 2-by-2 layout. Before running, "
        "predict which prior will be narrowed most strongly by the likelihood.",
        """KNOWN_RESPONSE = {
    "H1": dict(gain=1.0 + 0.0j, delay=0.004),
    "L1": dict(gain=0.82 * np.exp(0.35j), delay=-0.006),
}
KNOWN_MASS_RATIOS = {"event_a": 0.85, "event_b": 1.0}
KNOWN_AMPLITUDES = {
    "event_a": 1.7273387669253173e-21,
    "event_b": 9.826371643184475e-22,
}
KNOWN_PHASES = {"event_a": 0.0, "event_b": 0.0}

ordered_candidates = sorted(unique_candidates, key=lambda candidate: candidate[1])
assert len(ordered_candidates) == 2, "Resolve the coincidence table before PE."
candidate_specs = []
for index, (bank_name, candidate_time, h1_peak, l1_peak, rank) in enumerate(ordered_candidates):
    label = f"event_{'ab'[index]}"
    candidate_specs.append(dict(
        label=label,
        time=candidate_time,
        chirp_bounds=((16.5, 26.0), (26.6, 41.0))[index],
        mass_ratio=KNOWN_MASS_RATIOS[label],
        amplitude=KNOWN_AMPLITUDES[label],
        phase=KNOWN_PHASES[label],
        search_mass=best_mass[bank_name]["H1"][h1_peak],
    ))

challenge_priors = {
    spec["label"]: dict(
        chirp_mass=spec["chirp_bounds"],
        geocent_time=(spec["time"] - 0.05, spec["time"] + 0.05),
    )
    for spec in candidate_specs
}
prior_rng = np.random.default_rng(20260830)
fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
for row, spec in enumerate(candidate_specs):
    for axis, parameter in zip(axes[row], ("chirp_mass", "geocent_time")):
        bounds = challenge_priors[spec["label"]][parameter]
        axis.hist(prior_rng.uniform(*bounds, 4000), bins=35,
                  density=True, histtype="step")
        axis.set(xlabel=parameter, ylabel="density")
    axes[row, 0].set_title(spec["label"])
fig.suptitle("Priors before sampling")
fig.tight_layout()
plt.show()""",
    ),
    md(
        r"""### Supplied two-parameter likelihood

For fixed amplitude, phase, mass ratio, and detector response, the only varying
parameters are $\theta=(\mathcal M,t_c)$. The supplied class evaluates the
Gaussian log-likelihood ratio

$$
\log \Lambda(\theta)=(d|h_\theta)-\frac{1}{2}(h_\theta|h_\theta),
\qquad
(a|b)=4\,\mathrm{Re}\sum_f
\frac{a^*(f)b(f)}{S_n(f)}\,\Delta f.
$$

Read the class from top to bottom and locate: (1) the eight-second data segment,
(2) the fixed quantities, (3) where chirp mass changes the waveform, and (4)
where coalescence time enters. You do not need to rewrite this class."""
    ),
    *challenge_task(
        7,
        "Run the sampler and check the result",
        "Analyse an eight-second segment around each coincident trigger with "
        "the supplied Newtonian inspiral model, response, mass ratio, amplitude, "
        "and phase. Sample only chirp mass and coalescence time with an ordinary "
        "Gaussian-noise likelihood. The likelihood and Metropolis sampler are "
        "supplied: identify where the two sampled parameters enter, then run "
        "them. Report medians and symmetric 90% intervals. **Suggested plots:** "
        "two-chain traces, chirp-mass marginals, the chirp-mass/time joint "
        "posterior, and whitened data versus residual around each merger.",
        """PE_DURATION = 8.0


class TwoParameterCBCLikelihood:
    '''Supplied Gaussian likelihood in chirp mass and geocentric time only.'''

    def __init__(self, specification):
        self.parameters = {"chirp_mass": None, "geocent_time": None}
        self.specification = specification
        requested_start = specification["time"] - 6.0
        first = int(round((requested_start - start_time) * sampling_frequency))
        self.segment_start = start_time + first / sampling_frequency
        count = int(PE_DURATION * sampling_frequency)
        self.frequency = np.fft.rfftfreq(count, 1 / sampling_frequency)
        self.frequency_spacing = 1 / PE_DURATION
        self.data_frequency = {
            detector: np.fft.rfft(strain[detector][first : first + count])
            / sampling_frequency
            for detector in ("H1", "L1")
        }
        self.psd = {
            detector: np.interp(self.frequency, *final_psd[detector])
            for detector in ("H1", "L1")
        }
        self.usable = (self.frequency >= 20) & (self.frequency <= 400)

    def detector_templates(self, parameters):
        local_time = parameters["geocent_time"] - self.segment_start
        return {
            detector: self.specification["amplitude"]
            * np.exp(1j * self.specification["phase"])
            * response["gain"]
            * newtonian_chirp(
                self.frequency,
                parameters["chirp_mass"],
                self.specification["mass_ratio"],
                local_time + response["delay"],
            )
            for detector, response in KNOWN_RESPONSE.items()
        }

    def log_likelihood(self, parameters):
        log_likelihood = 0.0
        for detector, template in self.detector_templates(parameters).items():
            data = self.data_frequency[detector]
            psd = self.psd[detector]
            overlap = 4 * self.frequency_spacing * np.real(np.sum(
                np.conj(template[self.usable]) * data[self.usable]
                / psd[self.usable]
            ))
            norm = 4 * self.frequency_spacing * np.sum(
                np.abs(template[self.usable]) ** 2 / psd[self.usable]
            )
            log_likelihood += overlap - 0.5 * norm
        return float(log_likelihood)


def log_posterior(likelihood, specification, state):
    chirp_mass, geocent_time = state
    mass_low, mass_high = specification["chirp_bounds"]
    if not (mass_low <= chirp_mass <= mass_high):
        return -np.inf
    if not (specification["time"] - 0.05 <= geocent_time
            <= specification["time"] + 0.05):
        return -np.inf
    return likelihood.log_likelihood(dict(
        chirp_mass=float(chirp_mass), geocent_time=float(geocent_time)
    ))


def run_metropolis(likelihood, specification, seed, proposal_scale,
                   steps=6000, chains=2):
    '''Supplied random-walk sampler; returns chains and acceptance fractions.'''
    sampler_rng = np.random.default_rng(20260831)
    output, acceptance = [], []
    for chain_index in range(chains):
        state = np.asarray(seed, dtype=float) + sampler_rng.normal(
            scale=0.2 * proposal_scale
        )
        current_logp = log_posterior(likelihood, specification, state)
        chain = np.empty((steps, 2))
        accepted = 0
        for step_index in range(steps):
            proposal = state + sampler_rng.normal(scale=proposal_scale)
            proposal_logp = log_posterior(likelihood, specification, proposal)
            if np.log(sampler_rng.random()) < proposal_logp - current_logp:
                state, current_logp = proposal, proposal_logp
                accepted += 1
            chain[step_index] = state
        output.append(chain)
        acceptance.append(accepted / steps)
    return output, acceptance


challenge_likelihoods = {
    spec["label"]: TwoParameterCBCLikelihood(spec) for spec in candidate_specs
}
challenge_chains, challenge_samples = {}, {}
for spec in candidate_specs:
    label = spec["label"]
    proposal_scale = np.array([
        0.025 if label == "event_a" else 0.06, 0.00025
    ])
    chains, acceptance = run_metropolis(
        challenge_likelihoods[label], spec,
        seed=(spec["search_mass"], spec["time"]),
        proposal_scale=proposal_scale,
    )
    challenge_chains[label] = chains
    challenge_samples[label] = np.concatenate([chain[1000:] for chain in chains])
    print(label, "acceptance fractions:", np.round(acceptance, 3))

posterior_parameters = ["chirp_mass", "geocent_time"]
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for column, spec in enumerate(candidate_specs):
    label = spec["label"]
    posterior = challenge_samples[label]
    print(f"\\n{label}")
    for parameter_index, parameter in enumerate(posterior_parameters):
        low, median, high = np.quantile(
            posterior[:, parameter_index], [0.05, 0.5, 0.95]
        )
        print(f"  {parameter:15s}: {median:.6f} [{low:.6f}, {high:.6f}]")
    mass_quantiles = np.quantile(posterior[:, 0], [0.05, 0.5, 0.95])
    time_offsets_ms = 1000 * (posterior[:, 1] - spec["time"])
    axes[0, column].hist(posterior[:, 0], bins=40, density=True, histtype="step")
    axes[0, column].axvspan(
        mass_quantiles[0], mass_quantiles[2], color="C0", alpha=0.15
    )
    axes[0, column].set(title=label, xlabel=r"$\\mathcal{M}$ [$M_\\odot$]",
                        ylabel="density")
    axes[1, column].hexbin(
        posterior[:, 0], time_offsets_ms, gridsize=38, mincnt=1, cmap="Blues"
    )
    axes[1, column].set(xlabel=r"$\\mathcal{M}$ [$M_\\odot$]",
                        ylabel=r"$t_c-t_{\\rm search}$ [ms]")
fig.suptitle("Two-parameter conditional posteriors")
fig.tight_layout()
plt.show()

# Trace plots: both chains should overlap after burn-in and look stationary.
for spec in candidate_specs:
    fig, trace_axes = plt.subplots(2, 1, figsize=(8, 3.2), sharex=True)
    for chain in challenge_chains[spec["label"]]:
        trace_axes[0].plot(chain[:, 0], lw=0.35)
        trace_axes[1].plot(1000 * (chain[:, 1] - spec["time"]), lw=0.35)
    trace_axes[0].set(ylabel="chirp mass")
    trace_axes[1].set(xlabel="step", ylabel="time offset [ms]")
    fig.suptitle(f"{spec['label']}: convergence check")
    plt.show()

fig, residual_axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
for column, spec in enumerate(candidate_specs):
    label = spec["label"]
    posterior = challenge_samples[label]
    median_parameters = dict(zip(posterior_parameters, np.median(posterior, axis=0)))
    likelihood = challenge_likelihoods[label]
    templates = likelihood.detector_templates(median_parameters)
    for row, detector in enumerate(("H1", "L1")):
        data_fd = likelihood.data_frequency[detector]
        residual_fd = data_fd - templates[detector]
        whitened_data_fd = np.zeros_like(data_fd)
        whitened_residual_fd = np.zeros_like(residual_fd)
        whitened_data_fd[likelihood.usable] = (
            data_fd[likelihood.usable]
            / np.sqrt(likelihood.psd[detector][likelihood.usable])
        )
        whitened_residual_fd[likelihood.usable] = (
            residual_fd[likelihood.usable]
            / np.sqrt(likelihood.psd[detector][likelihood.usable])
        )
        count = int(PE_DURATION * sampling_frequency)
        whitened_data = np.fft.irfft(whitened_data_fd, n=count)
        whitened_residual = np.fft.irfft(whitened_residual_fd, n=count)
        scale = np.std(whitened_data[: 3 * sampling_frequency])
        local_time = likelihood.segment_start + np.arange(count) / sampling_frequency - spec["time"]
        residual_axes[row, column].plot(
            local_time, whitened_data / scale, color="0.65", lw=0.5, label="data"
        )
        residual_axes[row, column].plot(
            local_time, whitened_residual / scale, color="C3", lw=0.6,
            label="residual",
        )
        residual_axes[row, column].set(
            xlim=(-1, 0.15), ylabel=f"{detector} whitened",
            title=label if row == 0 else None,
        )
        residual_axes[row, column].legend(fontsize=7)
residual_axes[-1, 0].set_xlabel("time from candidate [s]")
residual_axes[-1, 1].set_xlabel("time from candidate [s]")
fig.suptitle("Posterior-median residual check")
fig.tight_layout()
plt.show()

print("Interpretation prompt: does coherent chirp-like structure remain in either")
print("residual, and what would that imply about the model?")""",
    ),
    md(
        """## Submission checklist

- candidate geocentric times and H1/L1 peak SNRs;
- duration calculation from each bank's lower chirp-mass edge;
- treatment of the unmatched transient;
- final H1/L1 ASD plot and PSD-segment justification;
- two-dimensional prior plot;
- sampler acceptance fractions and a convergence comment;
- posterior corner plots, waveform overlays, and residual checks.

Do not claim an astrophysical detection probability or false-alarm rate from
this classroom bank."""
    ),
]
write(
    "09_lvk_blind_data_challenge.ipynb",
    "Supplement: blind LVK data challenge",
    student_challenge_cells,
)


answer_challenge_cells = [
    _module_intro(
        "Worked instructor analysis of the blind H1/L1 challenge.",
        "Run the search and coincidence sections first; unblind only in the "
        "last section.",
        "The search statistic is illustrative and uncalibrated. The PE uses a "
        "restricted Newtonian-inspiral, fixed-response and fixed-mass-ratio model. "
        "Only chirp mass and coalescence time are sampled with fast workshop "
        "Dynesty settings.",
    ),
    md(
        """## Data and supplied ranges

This answer deliberately follows the same order as the student worksheet.
The generating PSD and injection parameters are not read from the HDF5 file.

[Download the HDF5 data directly](../assets/lvk_blind_challenge.h5)."""
    ),
    *[deepcopy(cell) for cell in LVK_CHALLENGE_DATA_SETUP],
    md("""## 1. Inspect the data

The raw overview is a data-integrity check, not a detection plot."""),
    code(
        """step = sampling_frequency // 8
fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
for axis, detector in zip(axes, ("H1", "L1")):
    axis.plot(time[::step], strain[detector][::step], lw=0.35)
    axis.set(ylabel=f"{detector} strain")
axes[-1].set_xlabel("time after file start [s]")
fig.suptitle("Blind strain overview: no CBC is visible by eye")
plt.show()""",
        figure="lvk-challenge-overview",
    ),
    md(
        r"""## 2. Duration from the supplied prior

At leading order,

$$
t(f_\mathrm{low})=\frac{5}{256}
\left(\frac{G\mathcal M}{c^3}\right)^{-5/3}
(\pi f_\mathrm{low})^{-8/3}.
$$

The lower edge, not the unknown injected value, sets the longest template."""
    ),
    code(
        r"""MTSUN_SI = 4.925490947e-6


def inspiral_time(chirp_mass, f_low=20.0):
    return (
        5
        / 256
        * (MTSUN_SI * chirp_mass) ** (-5 / 3)
        * (np.pi * f_low) ** (-8 / 3)
    )


for label, lower_edge in (("bank A", 16.5), ("bank B", 26.6)):
    print(f"{label}: t20 at lower edge = {inspiral_time(lower_edge):.3f} s")
ANALYSIS_DURATION = 8.0
print("Use an 8 s PE segment: 6 s before and 2 s after coalescence.")"""
    ),
    md(
        """## 3. Robust initial PSD

For discovery only, median Welch averaging over the full record is robust to a
few short signals/transients. After locating candidates we replace it with a
clean off-source estimate."""
    ),
    code(
        """WELCH_SECONDS = 4


def median_welch(values):
    return welch(
        values,
        fs=sampling_frequency,
        window=("tukey", 0.2),
        nperseg=WELCH_SECONDS * sampling_frequency,
        noverlap=WELCH_SECONDS * sampling_frequency // 2,
        detrend=False,
        average="median",
    )


initial_psd = {detector: median_welch(values) for detector, values in strain.items()}
fig, ax = plt.subplots(figsize=(9, 4))
for detector, (frequency_welch, psd_welch) in initial_psd.items():
    usable = (frequency_welch >= 15) & (frequency_welch <= 450)
    ax.loglog(frequency_welch[usable], np.sqrt(psd_welch[usable]), label=detector)
ax.set(xlabel="frequency [Hz]", ylabel=r"ASD [1/$\\sqrt{\\mathrm{Hz}}$]", title="Initial line-free median-Welch estimates")
ax.legend()
plt.show()""",
        figure="lvk-challenge-initial-psd",
    ),
    md(
        r"""## 4. Coarse matched-filter search

We use a single representative mass ratio, $q=0.9$, and maximise over the
listed chirp masses. The template is the Newtonian stationary-phase inspiral

$$
\tilde h(f)\propto f^{-7/6}\exp\left[i\left(-\frac{\pi}{4}
+\frac{3}{128}(\pi\mathcal M_{\rm sec}f)^{-5/3}-2\pi f t_c\right)\right],
$$

with a smooth taper at the mass-dependent ISCO. Chirp mass and time therefore
retain their physical meanings, but merger/ringdown and higher-order phase are
absent. This is enough to locate the controlled signals; it is not a production
template bank, realistic LVK BBH model, or calibrated search."""
    ),
    code(
        """n_samples = len(time)
frequency = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)
frequency_spacing = 1 / duration
data_frequency = {
    detector: np.fft.rfft(values) / sampling_frequency
    for detector, values in strain.items()
}
psd_on_search_grid = {
    detector: np.interp(frequency, *initial_psd[detector])
    for detector in ("H1", "L1")
}

MTSUN_SI = 4.925490947e-6


def component_masses(chirp_mass, mass_ratio):
    eta = mass_ratio / (1 + mass_ratio) ** 2
    total_mass = chirp_mass / eta ** (3 / 5)
    primary_mass = total_mass / (1 + mass_ratio)
    return primary_mass, mass_ratio * primary_mass


def newtonian_chirp(frequency_array, chirp_mass, mass_ratio, coalescence_time=0.0):
    '''Newtonian SPA inspiral with an explicit smooth ISCO cutoff.'''
    waveform = np.zeros(frequency_array.size, dtype=complex)
    primary_mass, secondary_mass = component_masses(chirp_mass, mass_ratio)
    f_isco = 1 / (
        6 ** 1.5 * np.pi * MTSUN_SI * (primary_mass + secondary_mass)
    )
    taper_start = 0.85 * f_isco
    usable = (frequency_array >= 20.0) & (frequency_array < f_isco)
    taper = np.ones(frequency_array.size)
    taper_region = (frequency_array >= taper_start) & (frequency_array < f_isco)
    taper[taper_region] = 0.5 * (
        1
        + np.cos(
            np.pi
            * (frequency_array[taper_region] - taper_start)
            / (f_isco - taper_start)
        )
    )
    phase = (
        -np.pi / 4
        + 3
        / 128
        * (np.pi * MTSUN_SI * chirp_mass * frequency_array[usable]) ** (-5 / 3)
        - 2 * np.pi * frequency_array[usable] * coalescence_time
    )
    waveform[usable] = (
        frequency_array[usable] ** (-7 / 6)
        * taper[usable]
        * np.exp(1j * phase)
    )
    return waveform


def matched_filter_snr(data_fd, template_fd, psd):
    usable = (frequency >= 20) & (frequency <= 400) & np.isfinite(psd) & (psd > 0)
    integrand = np.zeros(frequency.size, dtype=complex)
    integrand[usable] = data_fd[usable] * np.conj(template_fd[usable]) / psd[usable]
    padded = np.zeros(n_samples, dtype=complex)
    padded[: frequency.size] = integrand
    correlation = 4 * frequency_spacing * n_samples * np.fft.ifft(padded)
    norm = np.sqrt(
        4
        * frequency_spacing
        * np.sum(np.abs(template_fd[usable]) ** 2 / psd[usable])
    )
    return np.abs(correlation) / norm


banks = {
    "A: 16.5--26": np.linspace(16.5, 26.0, 20),
    "B: 26.6--41": np.linspace(26.6, 41.0, 24),
}
search_snr = {
    bank_name: {detector: np.zeros(n_samples) for detector in ("H1", "L1")}
    for bank_name in banks
}
best_mass = {
    bank_name: {detector: np.zeros(n_samples) for detector in ("H1", "L1")}
    for bank_name in banks
}
for bank_name, chirp_masses in banks.items():
    for chirp_mass in chirp_masses:
        template = newtonian_chirp(frequency, chirp_mass, mass_ratio=0.9)
        for detector in ("H1", "L1"):
            trial = matched_filter_snr(
                data_frequency[detector], template, psd_on_search_grid[detector]
            )
            better = trial > search_snr[bank_name][detector]
            search_snr[bank_name][detector][better] = trial[better]
            best_mass[bank_name][detector][better] = chirp_mass

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for axis, bank_name in zip(axes, banks):
    for detector in ("H1", "L1"):
        axis.plot(time, search_snr[bank_name][detector], lw=0.8, label=detector)
    axis.axhline(6, color="k", ls="--", lw=0.8)
    axis.set(ylabel="bank-max SNR", title=bank_name, xlim=(145, 235), ylim=(0, None))
    axis.legend()
axes[-1].set_xlabel("time after file start [s]")
plt.show()""",
        figure="lvk-challenge-search",
    ),
    md(
        """## 5. Coincidence and the glitch

A loud H1 trigger near 190 s lacks an L1 partner. Both banks instead contain
time-consistent H1/L1 peaks near the two CBC candidates. A 20 ms window is
generous relative to the inter-site light-travel time."""
    ),
    code(
        """SNR_THRESHOLD = 6.0
COINCIDENCE_WINDOW = 0.020
peak_tables = {}
coincident_candidates = []
for bank_name in banks:
    peak_tables[bank_name] = {}
    for detector in ("H1", "L1"):
        peaks, properties = find_peaks(
            search_snr[bank_name][detector],
            height=SNR_THRESHOLD,
            distance=int(0.5 * sampling_frequency),
        )
        peak_tables[bank_name][detector] = peaks
        print(f"\\n{bank_name} {detector} peaks")
        for peak in peaks:
            print(
                f"  t={time[peak]:8.3f} s  rho={search_snr[bank_name][detector][peak]:5.1f}  "
                f"Mc~{best_mass[bank_name][detector][peak]:5.1f}"
            )
    for h1_peak in peak_tables[bank_name]["H1"]:
        separations = np.abs(time[peak_tables[bank_name]["L1"]] - time[h1_peak])
        if len(separations) and separations.min() <= COINCIDENCE_WINDOW:
            l1_peak = peak_tables[bank_name]["L1"][np.argmin(separations)]
            geocent_guess = 0.5 * (time[h1_peak] + time[l1_peak])
            coincident_candidates.append((bank_name, geocent_guess, h1_peak, l1_peak))

print("\\nCoincident candidates")
for bank_name, candidate_time, h1_peak, l1_peak in coincident_candidates:
    print(
        f"  {bank_name}: t~{candidate_time:.3f} s, "
        f"H1/L1 SNR={search_snr[bank_name]['H1'][h1_peak]:.1f}/"
        f"{search_snr[bank_name]['L1'][l1_peak]:.1f}"
    )

# One physical event can trigger both coarse banks. Cluster coincidences in
# time, then retain the bank with the larger quadrature-summed detector SNR.
unique_candidates = []
for candidate in sorted(coincident_candidates, key=lambda item: item[1]):
    bank_name, candidate_time, h1_peak, l1_peak = candidate
    rank = np.hypot(
        search_snr[bank_name]["H1"][h1_peak],
        search_snr[bank_name]["L1"][l1_peak],
    )
    if unique_candidates and abs(candidate_time - unique_candidates[-1][1]) < 0.2:
        if rank > unique_candidates[-1][-1]:
            unique_candidates[-1] = (*candidate, rank)
    else:
        unique_candidates.append((*candidate, rank))

print("\\nTwo time-clustered candidates")
for bank_name, candidate_time, h1_peak, l1_peak, rank in unique_candidates:
    print(f"  {bank_name}: t~{candidate_time:.3f} s, network rank={rank:.1f}")"""
    ),
    code(
        """transient_window = (time >= 185) & (time <= 195)
sos = butter(4, (20, 250), btype="bandpass", fs=sampling_frequency, output="sos")
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
for axis, detector in zip(axes, ("H1", "L1")):
    filtered = sosfiltfilt(sos, strain[detector][transient_window])
    filtered /= np.std(filtered)
    spec_frequency, spec_time, power = spectrogram(
        filtered,
        fs=sampling_frequency,
        window=("tukey", 0.2),
        nperseg=128,
        noverlap=112,
        mode="psd",
    )
    band = (spec_frequency >= 20) & (spec_frequency <= 250)
    image = axis.pcolormesh(
        185 + spec_time,
        spec_frequency[band],
        10 * np.log10(power[band] + 1e-12),
        shading="auto",
    )
    axis.set(ylabel="frequency [Hz]", title=detector)
axes[-1].set_xlabel("time after file start [s]")
fig.colorbar(image, ax=axes, label="relative power [dB]")
fig.suptitle("The 190 s transient is confined to H1")
plt.show()""",
        figure="lvk-challenge-glitch",
    ),
    md(
        """The answer is to veto the short interval around 190 s from this
search and exclude it from PSD estimation. Because it is far from both event
segments, no subtraction is needed. If continuous filtering through the
interval were required, use a tapered gate or validated inpainting rather than
an abrupt rectangular zero."""
    ),
    md("""## 6. Final off-source Welch PSDs

The first 128 seconds are clean and precede the earliest candidate by more than
the allowed waveform duration. Four-second, 50%-overlapped segments give 63
median-averaged periodograms per detector."""),
    code(
        """clean = time < 128.0
final_psd = {
    detector: median_welch(values[clean]) for detector, values in strain.items()
}
number_of_averages = 1 + (clean.sum() - 4 * sampling_frequency) // (2 * sampling_frequency)
print("clean duration:", clean.sum() / sampling_frequency, "s")
print("overlapping periodograms:", number_of_averages)

fig, ax = plt.subplots(figsize=(9, 4))
for detector, (frequency_welch, psd_welch) in final_psd.items():
    usable = (frequency_welch >= 15) & (frequency_welch <= 450)
    ax.loglog(frequency_welch[usable], np.sqrt(psd_welch[usable]), label=detector)
ax.set(xlabel="frequency [Hz]", ylabel=r"ASD [1/$\\sqrt{\\mathrm{Hz}}$]", title="Final 128 s off-source PSD estimates")
ax.legend()
plt.show()""",
        figure="lvk-challenge-final-psd",
    ),
    md(
        """## 7. Restricted priors and likelihoods

For workshop speed, the response, mass ratio, zero spins, amplitude (hence
distance), and phase are supplied and fixed. We sample only chirp mass and
geocentric time with an ordinary Gaussian-noise likelihood. This is a proper
two-parameter posterior **conditional on those known quantities**, not a full
CBC posterior. An actual LVK analysis would infer distance, orientation, phase,
calibration, and waveform systematics."""
    ),
    code(
        r"""KNOWN_RESPONSE = {
    "H1": dict(gain=1.0 + 0.0j, delay=0.004),
    "L1": dict(gain=0.82 * np.exp(0.35j), delay=-0.006),
}
candidate_specs = [
    dict(
        label="event_a", time=160.0, chirp_bounds=(16.5, 26.0),
        mass_ratio=0.85, amplitude=1.7273387669253173e-21, phase=0.0,
        search_mass=22.0,
    ),
    dict(
        label="event_b", time=224.0, chirp_bounds=(26.6, 41.0),
        mass_ratio=1.00, amplitude=9.826371643184475e-22, phase=0.0,
        search_mass=31.0,
    ),
]


challenge_priors = {
    spec["label"]: dict(
        chirp_mass=spec["chirp_bounds"],
        geocent_time=(spec["time"] - 0.05, spec["time"] + 0.05),
    )
    for spec in candidate_specs
}
prior_rng = np.random.default_rng(20260830)
fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
for row, spec in enumerate(candidate_specs):
    draws = {
        parameter: prior_rng.uniform(*bounds, 4000)
        for parameter, bounds in challenge_priors[spec["label"]].items()
    }
    for axis, parameter in zip(axes[row], ("chirp_mass", "geocent_time")):
        axis.hist(draws[parameter], bins=35, density=True, histtype="step")
        axis.set(xlabel=parameter, ylabel="density" if parameter == "chirp_mass" else None)
    axes[row, 0].set_title(spec["label"])
fig.suptitle("Priors are plotted before seeing the posterior")
fig.tight_layout()
plt.show()""",
        figure="lvk-challenge-priors",
    ),
    code(
        """PE_DURATION = 8.0


class TwoParameterCBCLikelihood:
    '''Toy network likelihood in chirp mass and geocentric time only.'''

    def __init__(self, specification):
        self.parameters = {"chirp_mass": None, "geocent_time": None}
        self.specification = specification
        self.segment_start = specification["time"] - 6.0
        first = int(round((self.segment_start - start_time) * sampling_frequency))
        count = int(PE_DURATION * sampling_frequency)
        self.frequency = np.fft.rfftfreq(count, 1 / sampling_frequency)
        self.frequency_spacing = 1 / PE_DURATION
        self.data_frequency = {
            detector: np.fft.rfft(strain[detector][first : first + count])
            / sampling_frequency
            for detector in ("H1", "L1")
        }
        self.psd = {
            detector: np.interp(self.frequency, *final_psd[detector])
            for detector in ("H1", "L1")
        }
        self.usable = (self.frequency >= 20) & (self.frequency <= 400)

    def detector_templates(self, parameters):
        local_geocent_time = parameters["geocent_time"] - self.segment_start
        return {
            detector: self.specification["amplitude"]
            * np.exp(1j * self.specification["phase"])
            * response["gain"]
            * newtonian_chirp(
                self.frequency,
                parameters["chirp_mass"],
                self.specification["mass_ratio"],
                local_geocent_time + response["delay"],
            )
            for detector, response in KNOWN_RESPONSE.items()
        }

    def network_statistics(self, parameters):
        templates = self.detector_templates(parameters)
        overlap = 0.0
        norm = 0.0
        for detector, template in templates.items():
            overlap += 4 * self.frequency_spacing * np.real(
                np.sum(
                    np.conj(template[self.usable])
                    * self.data_frequency[detector][self.usable]
                    / self.psd[detector][self.usable]
                )
            )
            norm += (
                4
                * self.frequency_spacing
                * np.sum(
                    np.abs(template[self.usable]) ** 2
                    / self.psd[detector][self.usable]
                )
            )
        return overlap, norm, templates

    def log_likelihood(self):
        overlap, norm, _ = self.network_statistics(self.parameters)
        if not np.isfinite(norm) or norm <= 0:
            return -np.inf
        # The data-only Gaussian normalisation is constant in (Mc, tc), so the
        # likelihood ratio is (d|h) - 1/2 (h|h).
        return float(overlap - 0.5 * norm)


challenge_likelihoods = {
    spec["label"]: TwoParameterCBCLikelihood(spec) for spec in candidate_specs
}
print("Built two independent two-parameter toy likelihoods.")"""
    ),
    md(
        r"""## 8. Run a two-dimensional Metropolis sampler

Two independent random-walk chains sample only $(\mathcal M,t_c)$. The coarse
matched-filter estimates from Section 4 supply the starting masses. This matters
because fixing phase makes the likelihood much narrower than the phase-maximised
search statistic. Acceptance fractions and trace agreement are required checks;
this short classroom run is not an evidence calculation."""
    ),
    code(
        """sampler_rng = np.random.default_rng(20260831)
challenge_chains = {}
challenge_samples = {}


def log_posterior(likelihood, specification, state):
    chirp_mass, geocent_time = state
    low, high = specification["chirp_bounds"]
    if not (low <= chirp_mass <= high):
        return -np.inf
    if not (
        specification["time"] - 0.05
        <= geocent_time
        <= specification["time"] + 0.05
    ):
        return -np.inf
    likelihood.parameters.update(
        chirp_mass=float(chirp_mass), geocent_time=float(geocent_time)
    )
    return likelihood.log_likelihood()


for spec in candidate_specs:
    label = spec["label"]
    likelihood = challenge_likelihoods[label]
    seed = np.array([spec["search_mass"], spec["time"]])
    proposal_scale = np.array([
        0.025 if label == "event_a" else 0.06,
        0.00025,
    ])
    event_chains = []
    for chain_index in range(2):
        state = seed + sampler_rng.normal(scale=0.2 * proposal_scale)
        state[0] = np.clip(state[0], *spec["chirp_bounds"])
        state[1] = np.clip(
            state[1], spec["time"] - 0.05, spec["time"] + 0.05
        )
        current_logp = log_posterior(likelihood, spec, state)
        chain = np.empty((6000, 2))
        accepted = 0
        for step_index in range(len(chain)):
            proposal = state + sampler_rng.normal(scale=proposal_scale)
            proposal_logp = log_posterior(likelihood, spec, proposal)
            if np.log(sampler_rng.random()) < proposal_logp - current_logp:
                state, current_logp = proposal, proposal_logp
                accepted += 1
            chain[step_index] = state
        print(label, "chain", chain_index + 1, "acceptance", accepted / len(chain))
        event_chains.append(chain)
    challenge_chains[label] = event_chains
    challenge_samples[label] = np.concatenate([chain[1000:] for chain in event_chains])
    print(label, "retained samples:", len(challenge_samples[label]))"""
    ),
    code(
        """posterior_parameters = ["chirp_mass", "geocent_time"]
for spec in candidate_specs:
    label = spec["label"]
    posterior = challenge_samples[label]
    print(f"\\n{label}")
    for column, parameter in enumerate(posterior_parameters):
        low, median, high = np.quantile(posterior[:, column], [0.05, 0.5, 0.95])
        print(f"  {parameter:22s} {median:9.3f} [{low:9.3f}, {high:9.3f}]")
    fig, trace_axes = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
    for chain in challenge_chains[label]:
        trace_axes[0].plot(chain[:, 0], lw=0.35, alpha=0.75)
        trace_axes[1].plot(chain[:, 1] - spec["time"], lw=0.35, alpha=0.75)
    trace_axes[0].set(ylabel="chirp mass")
    trace_axes[1].set(xlabel="step", ylabel=r"$t_c-t_0$ [s]")
    fig.suptitle(f"{label}: two-chain trace check")
    plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for column, spec in enumerate(candidate_specs):
    label = spec["label"]
    posterior = challenge_samples[label]
    mass_quantiles = np.quantile(posterior[:, 0], [0.05, 0.5, 0.95])
    time_offsets_ms = 1000 * (posterior[:, 1] - spec["time"])
    time_quantiles = np.quantile(time_offsets_ms, [0.05, 0.5, 0.95])
    axes[0, column].hist(posterior[:, 0], bins=40, density=True, histtype="step")
    axes[0, column].axvspan(
        mass_quantiles[0], mass_quantiles[2], color="C0", alpha=0.15
    )
    axes[0, column].axvline(mass_quantiles[1], color="C0", ls="--")
    axes[0, column].set(
        xlabel=r"$\\mathcal{M}$ [$M_\\odot$]", ylabel="density", title=label
    )
    axes[1, column].hexbin(
        posterior[:, 0], time_offsets_ms, gridsize=38, mincnt=1, cmap="Blues"
    )
    axes[1, column].axvline(mass_quantiles[1], color="C3", ls="--", lw=1)
    axes[1, column].axhline(time_quantiles[1], color="C3", ls="--", lw=1)
    axes[1, column].set(
        xlabel=r"$\\mathcal{M}$ [$M_\\odot$]",
        ylabel=r"$t_c-t_0$ [ms]",
    )
fig.suptitle("Two-parameter conditional posteriors")
fig.tight_layout()
plt.show()""",
        figure="lvk-challenge-posteriors",
    ),
    md(
        """## 9. Waveform and residual checks

The posterior median should explain coherent power in both detectors. A corner
plot alone does not test whether a CBC waveform actually removed the candidate."""
    ),
    code(
        """fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex="col")
for column, spec in enumerate(candidate_specs):
    label = spec["label"]
    posterior = challenge_samples[label]
    median_parameters = dict(
        zip(posterior_parameters, np.median(posterior, axis=0))
    )
    likelihood = challenge_likelihoods[label]
    overlap, norm, templates = likelihood.network_statistics(median_parameters)
    for row, detector in enumerate(("H1", "L1")):
        model = templates[detector]
        residual = likelihood.data_frequency[detector] - model
        usable = likelihood.usable
        psd = likelihood.psd[detector]
        whitened_data_fd = np.zeros_like(likelihood.data_frequency[detector])
        whitened_residual_fd = np.zeros_like(residual)
        whitened_data_fd[usable] = (
            likelihood.data_frequency[detector][usable] / np.sqrt(psd[usable])
        )
        whitened_residual_fd[usable] = residual[usable] / np.sqrt(psd[usable])
        whitened_data = np.fft.irfft(
            whitened_data_fd, n=int(PE_DURATION * sampling_frequency)
        )
        whitened_residual = np.fft.irfft(
            whitened_residual_fd, n=int(PE_DURATION * sampling_frequency)
        )
        reference_scale = np.std(whitened_data[: 3 * sampling_frequency])
        whitened_data /= reference_scale
        whitened_residual /= reference_scale
        local_time = np.arange(len(whitened_data)) / sampling_frequency - 6
        axes[row, column].plot(
            local_time, whitened_data, color="0.65", lw=0.5, label="data"
        )
        axes[row, column].plot(
            local_time,
            whitened_residual,
            color="C3",
            lw=0.6,
            label="residual",
        )
        axes[row, column].set(
            xlim=(-1.0, 0.15),
            ylabel=f"{detector} whitened",
            title=label if row == 0 else None,
        )
        axes[row, column].legend(fontsize=7)
axes[-1, 0].set_xlabel("time from candidate [s]")
axes[-1, 1].set_xlabel("time from candidate [s]")
fig.suptitle("Median-model residual with fixed amplitude and phase")
fig.tight_layout()
plt.show()""",
        figure="lvk-challenge-residuals",
    ),
    md("""## 10. Unblind

Only now compare the recovered values with the data-generation truth."""),
    code(
        """truth = [
    dict(chirp_mass=22.0, mass_ratio=0.85, geocent_time=160.0, network_snr=30.0),
    dict(chirp_mass=31.0, mass_ratio=1.00, geocent_time=224.0, network_snr=15.0),
]
for event in truth:
    eta = event["mass_ratio"] / (1 + event["mass_ratio"]) ** 2
    total_mass = event["chirp_mass"] / eta ** (3 / 5)
    primary_mass = total_mass / (1 + event["mass_ratio"])
    secondary_mass = event["mass_ratio"] * primary_mass
    print(
        f"Mc={event['chirp_mass']:.1f}, q={event['mass_ratio']:.2f}, "
        f"m1={primary_mass:.2f}, m2={secondary_mass:.2f}, "
        f"t={event['geocent_time']:.1f} s, target network SNR={event['network_snr']:.0f}"
    )
print("H1-only sine-Gaussian: t=190.0 s, f0=75 Hz, target optimal SNR=45")"""
    ),
]
write(
    "09_lvk_blind_data_challenge_answer.ipynb",
    "Instructor answer: blind LVK data challenge",
    answer_challenge_cells,
    orphan=True,
)


# Remove the three transitional monoliths.  They are still generated above as
# the migration source, but they are not part of the public course artefacts.
for transitional_name in (
    "00_basics_parameter_estimation.ipynb",
    "01_lvk_compact_binary_parameter_estimation.ipynb",
    "02_lisa_parameter_estimation_and_global_fit.ipynb",
):
    (OUT / transitional_name).unlink()
