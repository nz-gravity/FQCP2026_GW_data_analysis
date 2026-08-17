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
6. replace white noise by a gravitational-wave-style PSD-weighted likelihood.

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
        md(r"""## 5. The gravitational-wave bridge: PSD and Whittle likelihood

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
            """## Checks and takeaways

1. Widen the prior: which marginal changes most?
2. Halve the assumed `sigma`: does the posterior become more accurate or merely more confident?
3. Why may the PSD-dependent likelihood normalisation be dropped for fixed-PSD
   PE but not when comparing noise models?
4. Change the prior width in the evidence cell. Why does the posterior near its
   peak barely move while the Bayes factor can change?

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
geometry, PSDs, projection, and injection. A readable one-dimensional posterior
replaces a slow live sampler. The aim is to make every layer visible before a
high-level library combines them."""),
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
        md("""## 3. Inject and infer manually

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
        md(
            """## 4. Why a network localises the sky

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
            r"""## 5. From events to a population

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

## 6. Bilby production composition

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
        md("""### Manual one-parameter likelihood

Perturb the source frequency, regenerate the moving-constellation response, and
compare optimal SNR with detected/matched SNR. A loud template can still match
the data poorly. The peak of this likelihood scan is the maximum-likelihood
frequency; normalising `exp(logL) × prior` would turn the grid into a
posterior."""),
        code(
            """offsets=np.linspace(-7e-7,7e-7,61); detected=[]; logL=[]; trial_templates=[]
for offset in offsets:
    trial=GBObject(f0=np.array([3e-3+offset]),fdot=np.array([1e-17]),A=np.array([2e-22]),ra=np.array([1.]),dec=np.array([.4]),psi=np.array([.3]),iota=np.array([.8]),phi0=np.array([.2]),t_init=0.)
    p=trial.to_jaxgb_array(t0=0); a,e,_=simulator.sum_tdi(p,int(simulator.get_kmin(parameters[:,0])[0]),int(simulator.get_kmin(parameters[:,0])[0])+simulator.n,tdi_generation=2,tdi_combination="AET")
    h=np.stack([np.asarray(a),np.asarray(e)]); trial_templates.append(h); rho=np.sqrt(inner(h,h)); detected.append(inner(template,h)/rho); logL.append(-.5*inner(template-h,template-h))
fig,axes=plt.subplots(1,2,figsize=(10,3.3)); axes[0].plot(offsets,detected); axes[0].axhline(optimal_snr,color="k",ls="--")
axes[0].set(xlabel="frequency offset [Hz]",ylabel="detected SNR",title="Match falls away from the signal")
axes[1].plot(offsets,np.array(logL)-np.max(logL)); axes[1].set(xlabel="frequency offset [Hz]",ylabel=r"$\\Delta \\log \\mathcal{L}$",title="Likelihood localises frequency"); plt.show()"""
        ),
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
