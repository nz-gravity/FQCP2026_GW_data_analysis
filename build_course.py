"""Build the Colab-first FQCP 2026 compact-binary PE mini-course."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks"


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)


def write(name, title, cells):
    header = md(f"""# {title}

**FQCP 2026 · Bayesian parameter estimation for compact binaries**

> Self-contained and designed for Google Colab. Run top to bottom; **Extension** cells may be skipped live.
""")
    nb = nbf.v4.new_notebook(cells=[header, *cells])
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": name, "provenance": []},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / name)
    print("Wrote", OUT / name)


STANDARD_SETUP = code('''import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
rng = np.random.default_rng(20260817)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["animation.html"] = "jshtml"
print("Running in Colab:", IN_COLAB)''')

HELPER_SETUP = code('''# One-file helper pattern: download from GitHub in Colab, import locally otherwise.
import sys
import urllib.request
from pathlib import Path

HELPER_URL = "https://raw.githubusercontent.com/nz-gravity/FQCP2026_GW_data_analysis/main/fqcp_helpers.py"
if IN_COLAB:
    urllib.request.urlretrieve(HELPER_URL, "fqcp_helpers.py")
else:
    candidates = [Path.cwd(), Path.cwd()/"FQCP2026_GW_data_analysis", Path.cwd().parent]
    helper_parent = next((path for path in candidates if (path/"fqcp_helpers.py").exists()), None)
    if helper_parent is None:
        raise FileNotFoundError("Could not locate fqcp_helpers.py")
    sys.path.insert(0, str(helper_parent))

from fqcp_helpers import equal_tailed_interval, frequency_inner_product, normalise_log_density
print("Course helpers loaded from", HELPER_URL if IN_COLAB else helper_parent)''')


write("00_start_here.ipynb", "Start here: how to use this mini-course", [
    md(r'''## Goal

By the end you should be able to:

1. explain a posterior as **prior × likelihood**, up to normalisation;
2. construct the PSD-weighted Gaussian/Whittle likelihood used in GW inference;
3. recognise how CBC parameters change a waveform and infer a chirp mass;
4. explain why population selection and LISA source overlap require larger models.

### Live route

Open notebooks `01`, `02`, and `04`. Notebook `03` is a short live section if time permits and a useful follow-up otherwise.

\[
\theta \longrightarrow h(f;\theta) \longrightarrow d(f)-h(f;\theta)
\longrightarrow p(d\mid\theta) \longrightarrow p(\theta\mid d).
\]

The same map appears in every chapter. Only the signal, noise model, dimension, and computational strategy change.'''),
    STANDARD_SETUP,
    md('''## Reusable helper file

Colab runtimes are temporary, but they can fetch a plain Python module from GitHub. The next cell downloads `fqcp_helpers.py` in Colab and imports the local copy during development. For a released course, replace `main` in the URL with a version tag such as `v1.0.0` so old notebooks remain reproducible.'''),
    HELPER_SETUP,
    code('''import scipy, matplotlib
print("NumPy", np.__version__)
print("SciPy", scipy.__version__)
print("Matplotlib", matplotlib.__version__)
assert tuple(map(int, np.__version__.split(".")[:2])) >= (1, 26)
print("Environment check passed.")'''),
    md('''## Vocabulary

| Term | Meaning |
| --- | --- |
| data, `d` | detector output: signal plus noise |
| parameters, `theta` | quantities we want to learn |
| prior | plausible values before this dataset |
| likelihood | compatibility of parameters with data and noise model |
| posterior | updated distribution after seeing the data |
| PSD | noise power as a function of frequency |
| evidence | probability of the data under a whole model |

<details><summary>Why not report only a best fit?</summary>

A best fit does not show uncertainty, degeneracies, multiple solutions, or prior sensitivity. A posterior can show all four.

</details>

## Before teaching

- Run all notebooks in a fresh Colab runtime.
- Open each notebook in a browser tab before the session.
- Animations use JavaScript HTML and do not need `ffmpeg`.
- Re-test the pinned ripple installation shortly before the workshop.'''),
])


write("01_bayes_and_whittle.ipynb", "1. Bayes, coloured noise, and the Whittle likelihood", [
    md(r'''## Goal

Turn `data = signal + noise` into a posterior. Frequency bins with large noise PSD should influence the fit less.

\[
\ln p(d\mid\theta)=-\frac12\langle d-h(\theta)\mid d-h(\theta)\rangle+C,
\qquad \langle a\mid b\rangle=4\,\mathrm{Re}\sum_k\frac{a_kb_k^*}{S_n(f_k)}\Delta f.
\]

We may omit `C` while PSD and data are fixed and only parameters change.'''),
    STANDARD_SETUP,
    code('''duration, sample_rate = 8.0, 256
time = np.arange(0, duration, 1/sample_rate)
frequency = np.fft.rfftfreq(time.size, 1/sample_rate)
df = frequency[1] - frequency[0]
psd = 0.15**2 * (1 + (25/np.maximum(frequency, 1))**4)

def signal(amplitude):
    return amplitude*np.sin(2*np.pi*18*time + 0.3)

def inner_product(a_f, b_f):
    return 4*np.real(np.sum(a_f*np.conj(b_f)/psd)*df)

def log_likelihood(data_f, model_f):
    residual = data_f-model_f
    return -0.5*inner_product(residual, residual)

true_amplitude = 0.8
# A chapter-local seed gives a representative draw whose 90% interval contains
# the injection; later discuss why coverage is a repeated-experiment property.
noise_rng = np.random.default_rng(20260812)
noise_f = np.fft.rfft(noise_rng.normal(size=time.size))*np.sqrt(psd)
data = signal(true_amplitude)+np.fft.irfft(noise_f, n=time.size)
data_f = np.fft.rfft(data)

fig, ax = plt.subplots(figsize=(7,3))
ax.loglog(frequency[1:], np.sqrt(psd[1:]))
ax.set(xlabel="frequency [Hz]", ylabel="noise ASD [toy units]", title="Coloured noise")
plt.show()'''),
    code('''amplitude_grid = np.linspace(-0.2, 1.8, 500)
log_posterior = np.array([log_likelihood(data_f, np.fft.rfft(signal(a))) for a in amplitude_grid])
posterior = np.exp(log_posterior-log_posterior.max())
posterior /= np.trapezoid(posterior, amplitude_grid)
cdf = np.r_[0, np.cumsum((posterior[:-1]+posterior[1:])*np.diff(amplitude_grid)/2)]
cdf /= cdf[-1]
interval = np.interp([.05,.95], cdf, amplitude_grid)

fig, ax = plt.subplots(figsize=(7,3))
ax.plot(amplitude_grid, posterior)
ax.axvline(true_amplitude, color="k", ls="--", label="injection")
ax.axvspan(*interval, alpha=.2, label="90% credible interval")
ax.set(xlabel="amplitude", ylabel="posterior density", title="Posterior, not just a best fit")
ax.legend(); plt.show()
print("90% interval:", np.round(interval,3))'''),
    md('''## Animation: information accumulates

Each frame uses a longer prefix of the same data. This illustrates posterior concentration; every frame starts from the same prior.'''),
    code('''frame_lengths = np.linspace(128, time.size, 30, dtype=int)
fig, (ax_d, ax_p) = plt.subplots(1,2,figsize=(10,3.2))
line_d, = ax_d.plot([],[],lw=.8); line_p, = ax_p.plot([],[],color="C1")
ax_d.set(xlim=(0,duration), ylim=(data.min(),data.max()), xlabel="time [s]", ylabel="data")
ax_p.set(xlim=(amplitude_grid.min(),amplitude_grid.max()), ylim=(0,posterior.max()*1.2), xlabel="amplitude", ylabel="density")
ax_p.axvline(true_amplitude,color="k",ls="--")

def update(frame):
    n = frame_lengths[frame]
    d_f = np.fft.rfft(data[:n]); f = np.fft.rfftfreq(n,1/sample_rate); d_fq=f[1]-f[0]
    s_n = 0.15**2*(1+(25/np.maximum(f,1))**4)
    values=[]
    for a in amplitude_grid:
        r=d_f-np.fft.rfft(signal(a)[:n])
        values.append(-2*d_fq*np.real(np.sum(r*np.conj(r)/s_n)))
    density=np.exp(values-np.max(values)); density/=np.trapezoid(density,amplitude_grid)
    line_d.set_data(time[:n],data[:n]); line_p.set_data(amplitude_grid,density)
    ax_d.set_title(f"data used: {time[n-1]:.1f} s")
    return line_d,line_p

animation = FuncAnimation(fig,update,frames=len(frame_lengths),interval=120)
plt.close(fig); display(HTML(animation.to_jshtml()))'''),
    md('''## Checks

1. What happens if the PSD is underestimated near the signal?
2. Why can `C` be ignored here but not always in noise-model comparison?

<details><summary>Answers</summary>

1. Those bins receive too much weight, so uncertainty can be underestimated and the fit can chase noise.
2. The omitted term is constant only while covariance/PSD and data are fixed.

</details>

## Next step

Replace the sine with a physical CBC waveform while retaining the same residual and PSD weighting.'''),
])


write("02_lvk_cbc_with_ripple.ipynb", "2. LVK compact-binary waveforms with rippleGW", [
    md('''## Goal

Use a real IMRPhenomD inspiral–merger–ringdown waveform inside a likelihood we can read. We infer one parameter live to understand likelihood geometry before a production-scale analysis.

This chapter pins `rippleGW==0.2.1`: rippleGW is pre-1.0, so unpinned teaching code may drift.'''),
    code('''import os, sys, subprocess, importlib.util
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
if importlib.util.find_spec("ripplegw") is None:
    if IN_COLAB:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","rippleGW==0.2.1"])
    else:
        raise ImportError("Install rippleGW==0.2.1, or run in Colab.")'''),
    code('''import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from jax import config, jit
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

rng=np.random.default_rng(20260817)
plt.style.use("seaborn-v0_8-whitegrid"); plt.rcParams["animation.html"]="jshtml"
frequency=jnp.linspace(20.,512.,985); df=float(frequency[1]-frequency[0]); f_ref=20.

def parameters(m1=36.,m2=29.,chi1=.1,chi2=-.1,distance=400.,tc=0.,phase=0.,inclination=.4):
    chirp_mass,eta=ms_to_Mc_eta(jnp.array([m1,m2]))
    return jnp.array([chirp_mass,eta,chi1,chi2,distance,tc,phase,inclination])

@jit
def waveform(theta):
    hp,_=gen_IMRPhenomD_hphc(frequency,theta,f_ref)
    return hp

theta_true=parameters(); h_true=waveform(theta_true)
print("chirp mass",round(float(theta_true[0]),3),"solar masses")
print("symmetric mass ratio",round(float(theta_true[1]),4))'''),
    code('''fig,axes=plt.subplots(1,2,figsize=(11,3.3))
axes[0].loglog(np.asarray(frequency),np.abs(np.asarray(h_true)))
axes[0].set(xlabel="frequency [Hz]",ylabel=r"$|h_+(f)|$",title="IMRPhenomD amplitude")
axes[1].plot(np.asarray(frequency),np.unwrap(np.angle(np.asarray(h_true))))
axes[1].set(xlabel="frequency [Hz]",ylabel="unwrapped phase [rad]",title="Phase carries mass information")
plt.show()'''),
    md('''## Animation: vary chirp mass

Mass ratio, spins, distance, time, phase, and inclination remain fixed. Chirp mass strongly changes phase evolution.'''),
    code('''mass_frames=np.linspace(18,42,30)
fig,ax=plt.subplots(figsize=(7,3.3)); line,=ax.loglog([],[])
ax.set(xlim=(20,512),ylim=(2e-25,2e-22),xlabel="frequency [Hz]",ylabel=r"$|h_+(f)|$")
def update_mass(frame):
    h=np.asarray(waveform(theta_true.at[0].set(mass_frames[frame])))
    line.set_data(np.asarray(frequency),np.abs(h)); ax.set_title(f"chirp mass = {mass_frames[frame]:.1f} solar masses")
    return (line,)
mass_animation=FuncAnimation(fig,update_mass,frames=len(mass_frames),interval=130)
plt.close(fig); display(HTML(mass_animation.to_jshtml()))'''),
    code('''# Simulated one-polarisation data and a toy LVK-like PSD
f=np.asarray(frequency)
psd=(2.2e-23)**2*((35/f)**4+1+(f/260)**2)
noise_scale=np.sqrt(psd/(4*df))
noise=noise_scale*(rng.normal(size=f.size)+1j*rng.normal(size=f.size))
data=np.asarray(h_true)+noise

def log_likelihood(model):
    residual=data-np.asarray(model)
    return -2*df*np.real(np.sum(residual*np.conj(residual)/psd))

mass_grid=np.linspace(float(theta_true[0])-5,float(theta_true[0])+5,220)
log_posterior=np.array([log_likelihood(waveform(theta_true.at[0].set(m))) for m in mass_grid])
posterior=np.exp(log_posterior-log_posterior.max()); posterior/=np.trapezoid(posterior,mass_grid)
fig,ax=plt.subplots(figsize=(7,3)); ax.plot(mass_grid,posterior)
ax.axvline(float(theta_true[0]),color="k",ls="--",label="injection")
ax.set(xlabel="chirp mass [solar masses]",ylabel="posterior density",title="rippleGW inside our Whittle likelihood")
ax.legend(); plt.show()'''),
    md('''## Boundary and next steps

rippleGW generated a genuine IMRPhenomD polarisation. The detector analysis is intentionally simplified: one polarisation, analytic PSD, no antenna response/calibration, and one free parameter.

<details><summary>What parameter should be freed next?</summary>

Coalescence time is a good teaching choice because it produces a linear frequency-domain phase shift and a clear correlation. Distance is simpler but mostly changes amplitude.

</details>

- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)
- [GWOSC tutorials](https://gwosc.org/tutorials/)'''),
])


write("03_population_inference.ipynb", "3. From individual binaries to a population", [
    md(r'''## Goal

See why the detected catalogue is not the astrophysical population.

\[
p(\Lambda\mid\{d_i\},\mathrm{det})\propto p(\Lambda)
\prod_i\frac{\int p(d_i\mid\theta)p(\theta\mid\Lambda)d\theta}{\alpha(\Lambda)}.
\]

For clarity, masses are treated as exactly measured and only a population mean is inferred.'''),
    STANDARD_SETUP,
    code('''from scipy.stats import norm
true_mean,true_width=28.,5.
all_masses=rng.normal(true_mean,true_width,5000)
all_masses=all_masses[(all_masses>8)&(all_masses<55)]
def p_detect(mass): return 1/(1+np.exp(-(mass-22)/3.5))
detected=all_masses[rng.random(all_masses.size)<p_detect(all_masses)][:30]
fig,axes=plt.subplots(1,2,figsize=(10,3.2)); grid=np.linspace(8,55)
axes[0].plot(grid,p_detect(grid)); axes[0].set(xlabel="mass",ylabel="detection probability",title="Toy selection")
axes[1].hist(all_masses,bins=35,density=True,histtype="step",lw=2,label="underlying")
axes[1].hist(detected,bins=12,density=True,alpha=.55,label="detected")
axes[1].set(xlabel="chirp mass [toy units]",ylabel="density",title="Observed is not underlying"); axes[1].legend(); plt.show()
print("underlying mean",round(all_masses.mean(),2),"detected mean",round(detected.mean(),2))'''),
    code('''mean_grid=np.linspace(15,40,350); integration_grid=np.linspace(8,55,800)
log_naive=[]; log_corrected=[]
for mean in mean_grid:
    event_term=norm.logpdf(detected,mean,true_width).sum()
    alpha=np.trapezoid(norm.pdf(integration_grid,mean,true_width)*p_detect(integration_grid),integration_grid)
    log_naive.append(event_term); log_corrected.append(event_term-len(detected)*np.log(alpha))
def normalise(log_density):
    density=np.exp(log_density-np.max(log_density)); return density/np.trapezoid(density,mean_grid)
fig,ax=plt.subplots(figsize=(7,3.2))
ax.plot(mean_grid,normalise(np.array(log_naive)),label="ignores selection")
ax.plot(mean_grid,normalise(np.array(log_corrected)),label="selection-aware")
ax.axvline(true_mean,color="k",ls="--",label="injection")
ax.set(xlabel="population mean [toy units]",ylabel="posterior density",title="Selection changes the answer"); ax.legend(); plt.show()'''),
    md('''## Check

<details><summary>Why divide by alpha once per event?</summary>

Conditioning each observed event on detection renormalises its population density by the detectable fraction. For a fixed observed count this contributes `alpha(Lambda)^(-N)`.

</details>

Real analyses integrate event likelihoods/posterior samples, infer several hyperparameters, estimate selection with injections, and often include the rate.'''),
])


write("04_lisa_global_fit.ipynb", "4. LISA and the idea of a global fit", [
    md('''## Goal

Understand why long-lived overlapping LISA sources should often be modelled jointly. Two nearby sinusoids in one channel provide a small visual analogy.'''),
    STANDARD_SETUP,
    code('''time=np.linspace(0,30,3000,endpoint=False)
f1,f2=.70,.76; phi1,phi2=.2,1.1; a1_true,a2_true=1.,.65; sigma=.55
def source_1(a): return a*np.sin(2*np.pi*f1*time+phi1)
def source_2(a): return a*np.sin(2*np.pi*f2*time+phi2)
signal_true=source_1(a1_true)+source_2(a2_true)
data=signal_true+rng.normal(0,sigma,time.size); show=time<10
fig,ax=plt.subplots(figsize=(8,3)); ax.plot(time[show],data[show],lw=.7,label="data")
ax.plot(time[show],signal_true[show],lw=2,label="two-source signal")
ax.set(xlabel="time [toy units]",ylabel="strain",title="Long-lived signals overlap"); ax.legend(); plt.show()'''),
    code('''grid=np.linspace(-.2,1.6,180); log_joint=np.empty((grid.size,grid.size))
for i,a1 in enumerate(grid):
    for j,a2 in enumerate(grid):
        residual=data-source_1(a1)-source_2(a2)
        log_joint[i,j]=-.5*np.sum((residual/sigma)**2)
joint=np.exp(log_joint-log_joint.max()); joint/=joint.sum()
marginal=joint.sum(axis=1); marginal/=np.trapezoid(marginal,grid)
wrong_log=np.array([-.5*np.sum(((data-source_1(a))/sigma)**2) for a in grid])
wrong=np.exp(wrong_log-wrong_log.max()); wrong/=np.trapezoid(wrong,grid)
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
axes[0].contourf(grid,grid,joint.T,levels=25,cmap="magma"); axes[0].plot(a1_true,a2_true,"c*",ms=11)
axes[0].set(xlabel="source 1 amplitude",ylabel="source 2 amplitude",title="Joint posterior")
axes[1].plot(grid,marginal,label="two-source fit"); axes[1].plot(grid,wrong,ls="--",label="source 2 omitted")
axes[1].axvline(a1_true,color="k",ls=":"); axes[1].set(xlabel="source 1 amplitude",ylabel="density",title="Incomplete models can bias"); axes[1].legend(); plt.show()'''),
    md('''## Animation: subtraction is only as good as the model

Watch the residual as the assumed source-2 amplitude approaches its true value.'''),
    code('''frames=np.linspace(0,a2_true,30); fig,ax=plt.subplots(figsize=(8,3.2)); line,=ax.plot([],[],lw=.8)
ax.axhline(0,color="k",lw=.7); ax.set(xlim=(0,10),ylim=(-2.5,2.5),xlabel="time",ylabel="data - model")
def update_subtraction(frame):
    residual=data-source_1(a1_true)-source_2(frames[frame])
    line.set_data(time[show],residual[show]); ax.set_title(f"assumed source-2 amplitude = {frames[frame]:.2f}")
    return (line,)
animation=FuncAnimation(fig,update_subtraction,frames=len(frames),interval=120)
plt.close(fig); display(HTML(animation.to_jshtml()))'''),
    md('''## Check and boundary

<details><summary>Why is this a miniature global fit?</summary>

Both source amplitudes are inferred simultaneously against one dataset. A realistic LISA fit may combine source classes, noise, backgrounds, multiple TDI channels, response models, and an unknown number of Galactic binaries.

</details>

Continue with LISA Analysis Tools Workshop Tutorial 1, Tutorial 6, and its mini-global-fit challenge.'''),
])
