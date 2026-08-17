"""Build the three Colab-first FQCP 2026 gravitational-wave PE notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks"

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

def write(name, title, cells):
    header = md(f"""# {title}

**FQCP 2026 · Bayesian parameter estimation for gravitational-wave sources**

> Google Colab worksheet for early-stage graduate students. Run from top to bottom; **Extension** sections may be skipped live.
""")
    notebook = nbf.v4.new_notebook(cells=[header, *cells])
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": name, "provenance": []},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUT / name)
    print("Wrote", OUT / name)

OUT.mkdir(parents=True, exist_ok=True)
for old_notebook in OUT.glob("*.ipynb"):
    old_notebook.unlink()

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


write("00_basics_parameter_estimation.ipynb", "Basics: what is parameter estimation?", [
    md(r'''## Goal

Parameter estimation (PE) means constructing a probability distribution for unknown parameters after observing data. We will follow the teaching sequence used in the NZ Bilby CBC workshop:

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

The evidence $\mathcal Z$ normalises the posterior and compares complete models. PE is not just optimisation: it retains uncertainty, correlations, multiple solutions, and prior dependence.'''),
    STANDARD_SETUP,
    md(r'''## 1. Data and a signal model

Assume $d_i=m t_i+c+n_i$ and independent Gaussian noise $n_i\sim\mathcal N(0,\sigma^2)$. Every likelihood statement is conditional on assumptions like these.'''),
    code('''true_parameters={"m":.5,"c":.2}; sigma=3.0
time=np.linspace(0,10,100)
def signal_model(time,m,c): return m*time+c
data=signal_model(time,**true_parameters)+rng.normal(0,sigma,time.size)
fig,ax=plt.subplots(figsize=(8,3.3)); ax.plot(time,data,"o",ms=3,label="data")
ax.plot(time,signal_model(time,**true_parameters),lw=2,label="injected signal")
ax.set(xlabel="time",ylabel="observation",title="Data = signal + noise"); ax.legend(); plt.show()'''),
    md(r'''## 2. Priors and prior predictive checks

Take $m\sim\mathrm{Uniform}(0,1.5)$ and $c\sim\mathrm{Uniform}(-5,5)$. A prior is part of the model, not an afterthought. Drawing curves from it checks whether the model can plausibly describe the data before inference.'''),
    code('''n_prior=2500
prior_m=rng.uniform(0,1.5,n_prior); prior_c=rng.uniform(-5,5,n_prior)
fig,axes=plt.subplots(1,2,figsize=(10,3.4))
axes[0].hist(prior_m,bins=30,density=True,histtype="step",label="m")
axes[0].hist(prior_c,bins=30,density=True,histtype="step",label="c")
axes[0].set(xlabel="parameter value",ylabel="prior density",title="Marginal priors"); axes[0].legend()
axes[1].plot(time,data,"o",ms=3,color="k")
for m,c in zip(prior_m[:40],prior_c[:40]): axes[1].plot(time,signal_model(time,m,c),color="C0",alpha=.08)
axes[1].set(xlabel="time",ylabel="observation",title="Prior predictive curves"); plt.show()'''),
    md(r'''## 3. Gaussian likelihood

\[
\log\mathcal L(d\mid m,c)=-\frac12\sum_i\left[
\frac{(d_i-mt_i-c)^2}{\sigma^2}+\log(2\pi\sigma^2)\right].
\]

Changing the assumed noise scale changes the width of the posterior. If the noise model is wrong, a mathematically correct sampler still gives a misleading answer.'''),
    code('''def log_likelihood(m,c):
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
axes[0].set_ylabel("intercept c"); plt.show()'''),
    md('''The posterior is a ridge: increasing the slope can be compensated by decreasing the intercept. Marginalisation integrates over the other parameter; it is not the same as holding it at a best-fit value.'''),
    code('''p_m=np.trapezoid(posterior,c_grid,axis=1); p_c=np.trapezoid(posterior,m_grid,axis=0)
def interval(grid,density):
    cdf=np.r_[0,np.cumsum((density[:-1]+density[1:])*np.diff(grid)/2)]; cdf/=cdf[-1]
    return np.interp([.05,.5,.95],cdf,grid)
fig,axes=plt.subplots(1,2,figsize=(9,3.2))
for ax,grid,density,name,truth in zip(axes,[m_grid,c_grid],[p_m,p_c],["m","c"],true_parameters.values()):
    q=interval(grid,density); ax.plot(grid,density); ax.axvline(truth,color="k",ls="--")
    ax.axvspan(q[0],q[2],alpha=.2); ax.set(xlabel=name,ylabel="marginal posterior",title=f"median {q[1]:.2f}; 90% [{q[0]:.2f}, {q[2]:.2f}]")
plt.show()'''),
    md('''## 4. Posterior predictive check

Draw parameter pairs from the posterior and map each through the signal model. This asks whether the inferred model can reproduce data like those observed.'''),
    code('''weights=(posterior/posterior.sum()).ravel(); choices=rng.choice(weights.size,size=250,replace=True,p=weights)
m_samples=M.ravel()[choices]; c_samples=C.ravel()[choices]
predictions=np.array([signal_model(time,m,c) for m,c in zip(m_samples,c_samples)])
low,median,high=np.quantile(predictions,[.05,.5,.95],axis=0)
fig,ax=plt.subplots(figsize=(8,3.3)); ax.plot(time,data,"o",ms=3,color="k",label="data")
ax.plot(time,median,label="posterior median"); ax.fill_between(time,low,high,alpha=.25,label="90% signal band")
ax.set(xlabel="time",ylabel="observation",title="Posterior predictive signal"); ax.legend(); plt.show()'''),
    md(r'''## 5. The gravitational-wave bridge: coloured noise

GW detector noise depends strongly on frequency. For approximately stationary Gaussian noise, positive-frequency bins give the Whittle likelihood

\[
\log\mathcal L=-\frac12(d-h\mid d-h)+C,\qquad
(a\mid b)=4\,\mathrm{Re}\sum_k\frac{\tilde a_k\tilde b_k^*}{S_n(f_k)}\Delta f.
\]

$S_n$ is the one-sided power spectral density (PSD); $\sqrt{S_n}$ is the amplitude spectral density (ASD). The PSD decides which residuals matter.'''),
    code('''from scipy.signal import welch
sample_rate=512; duration=32; t=np.arange(0,duration,1/sample_rate); f=np.fft.rfftfreq(t.size,1/sample_rate)
shape=1+(30/np.maximum(f,1))**4+(f/180)**2
coloured=np.fft.irfft(np.fft.rfft(rng.normal(size=t.size))*np.sqrt(shape),n=t.size)
f_psd,psd=welch(coloured,fs=sample_rate,nperseg=2048,average="median")
fig,ax=plt.subplots(figsize=(8,3.2)); ax.loglog(f_psd[1:],np.sqrt(psd[1:]))
ax.set(xlabel="frequency [Hz]",ylabel="ASD [toy units]",title="Estimate the noise before weighting residuals"); plt.show()'''),
    md('''## Checks and takeaways

1. Widen the prior: which marginal changes most?
2. Halve the assumed `sigma`: does the posterior become more accurate or merely more confident?
3. Why may the likelihood normalisation be dropped for fixed PSD PE but not when comparing noise models?

**Takeaway:** PE is a model–data–noise calculation. A posterior is only as trustworthy as the waveform, response, PSD, priors, and computation that define it.

Adapted from the local `nz_bilby_cbc_workshop_2024` and its source, Colm Talbot's Bayesian inference tutorial.'''),
])


write("01_lvk_compact_binary_parameter_estimation.ipynb", "LVK: compact-binary parameter estimation", [
    md(r'''## Goal and analysis map

Follow a compact version of the NZ Bilby workshop's full CBC flow:

\[
\theta_{\rm CBC}\rightarrow(h_+,h_\times)\rightarrow h_I
\rightarrow d_I=h_I+n_I\rightarrow\mathcal L_{\rm network}\rightarrow p(\theta\mid d).
\]

We use rippleGW for an actual IMRPhenomD waveform and Bilby for detector geometry, PSDs, projection, and injection. A readable one-dimensional posterior replaces a slow live sampler.'''),
    code('''import os,sys,subprocess,importlib.util
IN_COLAB="COLAB_RELEASE_TAG" in os.environ
missing=[p for p in ("ripplegw","bilby") if importlib.util.find_spec(p) is None]
if missing:
    if IN_COLAB: subprocess.check_call([sys.executable,"-m","pip","install","-q","rippleGW==0.2.1","bilby==2.8.0"])
    else: raise ImportError("Install rippleGW==0.2.1 and bilby==2.8.0, or run in Colab.")'''),
    code('''import logging
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
rng=np.random.default_rng(20260817)'''),
    md(r'''## 1. CBC parameters

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

Intrinsic/extrinsic is useful bookkeeping, but parameters remain correlated in the posterior.'''),
    code('''sample_rate,duration,f_min=1024,4,20.; gps_time=1126259462.4
frequency=np.fft.rfftfreq(int(sample_rate*duration),1/sample_rate); mask=frequency>=f_min; df=frequency[1]-frequency[0]
def ripple_parameters(chirp_mass=None,m1=36.,m2=29.,chi1=.1,chi2=-.1,distance=800.,tc=0.,phase=.3,inclination=.5):
    mc,eta=ms_to_Mc_eta(jnp.array([m1,m2])); mc=mc if chirp_mass is None else chirp_mass
    return jnp.array([mc,eta,chi1,chi2,distance,tc,phase,inclination])
def polarizations(theta):
    hp,hx=gen_IMRPhenomD_hphc(jnp.asarray(frequency[mask]),theta,f_min)
    result={"plus":np.zeros(frequency.size,dtype=complex),"cross":np.zeros(frequency.size,dtype=complex)}
    result["plus"][mask]=np.asarray(hp); result["cross"][mask]=np.asarray(hx); return result
theta_true=ripple_parameters(); injection_polarizations=polarizations(theta_true)
print(f"Detector-frame chirp mass: {float(theta_true[0]):.3f} solar masses")'''),
    code('''fig,axes=plt.subplots(1,2,figsize=(11,3.4))
for name,h in injection_polarizations.items(): axes[0].loglog(frequency[mask],np.abs(h[mask]),label=name)
axes[0].set(xlabel="frequency [Hz]",ylabel="strain / Hz",title="Radiation has two polarisations"); axes[0].legend()
axes[1].plot(frequency[mask],np.unwrap(np.angle(injection_polarizations["plus"][mask])))
axes[1].set(xlabel="frequency [Hz]",ylabel="phase [rad]",title="Chirp mass is measured mainly through phase"); plt.show()'''),
    md(r'''For a non-precessing circular binary, approximately
$h_+\propto(1+\cos^2\iota)/(2D_L)$ and $h_\times\propto\cos\iota/D_L$.
Inclination is the binary's orientation to us; polarisation angle rotates the plus/cross basis on the sky.'''),
    code('''frames=np.linspace(float(theta_true[0])-6,float(theta_true[0])+6,28)
fig,(aa,ap)=plt.subplots(1,2,figsize=(11,3.3)); la,=aa.loglog([],[]); lp,=ap.plot([],[])
aa.set(xlim=(20,512),ylim=(1e-25,3e-22),xlabel="frequency [Hz]",ylabel=r"$|h_+|$")
ap.set(xlim=(20,512),ylim=(-650,50),xlabel="frequency [Hz]",ylabel="relative phase [rad]")
def animate_mass(i):
    h=polarizations(theta_true.at[0].set(frames[i]))["plus"][mask]; phase=np.unwrap(np.angle(h)); phase-=phase[0]
    la.set_data(frequency[mask],np.abs(h)); lp.set_data(frequency[mask],phase); fig.suptitle(f"chirp mass = {frames[i]:.1f} solar masses"); return la,lp
animation=FuncAnimation(fig,animate_mass,frames=len(frames),interval=130); plt.close(fig); display(HTML(animation.to_jshtml()))'''),
    md(r'''## 2. From source to a detector network

For detector $I$,
\[
\tilde h_I=[F^I_+h_++F^I_\times h_\times]e^{-2\pi if\Delta t_I}.
\]
Bilby stores detector geometry and PSDs and applies this projection. This is the machinery that a high-level CBC likelihood normally hides.'''),
    code('''source_parameters=dict(ra=1.2,dec=-.4,psi=.7,geocent_time=gps_time)
ifos=bilby.gw.detector.InterferometerList(["H1","L1","V1"])
for ifo in ifos: ifo.set_strain_data_from_zero_noise(sampling_frequency=sample_rate,duration=duration,start_time=gps_time-2)
print("IFO     F+      Fx      delay [ms]")
for ifo in ifos:
    fp=ifo.antenna_response(source_parameters["ra"],source_parameters["dec"],gps_time,source_parameters["psi"],"plus")
    fx=ifo.antenna_response(source_parameters["ra"],source_parameters["dec"],gps_time,source_parameters["psi"],"cross")
    dt=ifo.time_delay_from_geocenter(source_parameters["ra"],source_parameters["dec"],gps_time)
    print(f"{ifo.name:>3}  {fp:+.3f}  {fx:+.3f}   {1e3*dt:+.2f}")'''),
    code('''fig,axes=plt.subplots(1,2,figsize=(11,3.4))
for ifo in ifos:
    asd=ifo.power_spectral_density.get_amplitude_spectral_density_array(frequency)
    axes[0].loglog(frequency[mask],asd[mask],label=ifo.name)
    response=ifo.get_detector_response(injection_polarizations,source_parameters,frequencies=frequency)
    axes[1].loglog(frequency[mask],np.abs(response[mask]),label=ifo.name)
axes[0].set(xlabel="frequency [Hz]",ylabel=r"ASD [1/$\\sqrt{\\mathrm{Hz}}$]",title="Each detector has a PSD")
axes[1].set(xlabel="frequency [Hz]",ylabel="projected strain / Hz",title="Each detector sees a different signal")
for ax in axes: ax.legend(); plt.show()'''),
    md('''## 3. Inject and infer

We use zero-noise data so the demonstration is deterministic: the data equal the injected signal, while the PSD still controls expected uncertainty. Replace `set_strain_data_from_zero_noise` with Bilby's PSD-noise method to study repeated noise realisations.'''),
    code('''for ifo in ifos: ifo.inject_signal_from_waveform_polarizations(source_parameters,injection_polarizations)
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
fig,ax=plt.subplots(figsize=(8,3.4)); ax.plot(mass_grid,density(np.array(logL_by_ifo["H1"])),label="H1 only")
ax.plot(mass_grid,density(logL_network),label="H1+L1+V1"); ax.axvline(float(theta_true[0]),color="k",ls="--",label="injection")
ax.set(xlabel="detector-frame chirp mass [solar masses]",ylabel="posterior density",title="A coherent network gives more information"); ax.legend(); plt.show()'''),
    md('''## 4. Why a network localises the sky

Timing alone gives a ring with two sites and smaller regions with three. Real Bilby localisation also uses coherent phase, antenna amplitudes, polarisation, distance–inclination correlations, waveform uncertainty, and sky priors.'''),
    code('''ra=np.linspace(-np.pi,np.pi,91); dec=np.linspace(-np.pi/2,np.pi/2,46); RA,DEC=np.meshgrid(ra,dec)
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
plt.show()'''),
    md('''## 5. Bilby production composition

```python
waveform_generator = bilby.gw.WaveformGenerator(...)
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator)
result = bilby.run_sampler(likelihood, priors, sampler="dynesty", ...)
```

Bilby then handles response projection, detector likelihoods, common parameters, priors, marginalisations, and sampler results. Yay Bilby.

## Boundary and extensions

- The live posterior frees only chirp mass; production BBH analyses may have about fifteen parameters plus nuisance/systematic choices.
- Design PSDs and zero noise are pedagogical. Real data contain PSD uncertainty, lines, glitches, non-stationarity, and calibration uncertainty.
- Use the NZ workshop's GW150914 section or [GWOSC tutorials](https://gwosc.org/tutorials/) as a real-data follow-up.

Adapted substantially from `nz_bilby_cbc_workshop_2024`, with its injection → PSD → prior → likelihood → result structure.'''),
])


write("02_lisa_parameter_estimation_and_global_fit.ipynb", "LISA: sensitivity, response, and the global fit", [
    md(r'''## Goal and analysis map

This notebook follows the LISA Analysis Tools Workshop progression:

\[
\text{sensitivity}\rightarrow\text{TDI data}\rightarrow(a\mid b)
\rightarrow\mathrm{SNR}\rightarrow\mathcal L\rightarrow
\text{single source}\rightarrow\text{unknown overlapping catalogue}.
\]

We use `lisatools` for LISA sensitivity curves and JaxGB for an actual moving-constellation Galactic-binary response. The final exercise is a miniature version of the LATW global-fit challenge.'''),
    code('''import os,sys,subprocess,importlib.util
IN_COLAB="COLAB_RELEASE_TAG" in os.environ
needed=("lisatools","gpubackendtools","jaxgb")
if any(importlib.util.find_spec(package) is None for package in needed):
    if IN_COLAB:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","lisaanalysistools==1.2.5","gpubackendtools==0.1.1","jaxgb==0.2.1","astropy==7.2.0"])
    else: raise ImportError("Install the pinned LISA requirements, or run in Colab.")'''),
    code('''import itertools
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML,display
from matplotlib.animation import FuncAnimation
from jax import config
config.update("jax_enable_x64",True)
from lisatools.sensitivity import A1TDISens,E1TDISens,SensitivityMatrix,get_sensitivity
from lisatools.utils.constants import YRSID_SI
from lisaorbits import EqualArmlengthOrbits
from jaxgb.jaxgb import JaxGB
from jaxgb.params import GBObject
rng=np.random.default_rng(20260817); plt.style.use("seaborn-v0_8-whitegrid"); plt.rcParams["animation.html"]="jshtml"'''),
    md(r'''## 1. LISA's band and source zoo

Ground-based detectors observe roughly tens of Hz to kHz. LISA targets approximately $10^{-4}$–$10^{-1}$ Hz, containing Galactic compact binaries, massive-black-hole binaries, EMRIs, stellar-origin binaries, stochastic backgrounds, and instrument noise. Long observations make many signals overlap.

Unlike a static right-angle detector, LISA is a heliocentric triangle that cartwheels as it orbits. Six delayed one-way laser links are combined into time-delay interferometry (TDI) variables. Orbital modulation helps localisation, while finite arms create a frequency-dependent response.'''),
    code('''year=YRSID_SI; AU=149597870700.; orbits=EqualArmlengthOrbits(); times=np.linspace(0,year,240)
positions=np.asarray(orbits.compute_position(times,[1,2,3])); fig,ax=plt.subplots(figsize=(5.4,5.4))
for i,label in enumerate(["spacecraft 1","spacecraft 2","spacecraft 3"]): ax.plot(positions[:,i,0]/AU,positions[:,i,1]/AU,label=label)
ax.plot(0,0,"o",color="gold",mec="k",label="Sun"); ax.set(xlabel="heliocentric x [AU]",ylabel="heliocentric y [AU]",title="An explicit LISA orbit model",aspect="equal"); ax.legend(); plt.show()'''),
    md('''## 2. Sensitivity and Galactic confusion

As in LATW Tutorial 1, start with the noise model. The unresolved Galactic foreground changes with observing time because longer data resolve and subtract more binaries.'''),
    code('''f_curve=np.logspace(-5,-1,1800)
instrument=SensitivityMatrix(f_curve,[A1TDISens,E1TDISens])
one_year=SensitivityMatrix(f_curve,[A1TDISens,E1TDISens],stochastic_params=(1*year,))
four_year=SensitivityMatrix(f_curve,[A1TDISens,E1TDISens],stochastic_params=(4*year,))
fig,ax=plt.subplots(figsize=(8,3.6))
ax.loglog(f_curve,np.sqrt(instrument.sens_mat[0]),label="instrument only")
ax.loglog(f_curve,np.sqrt(one_year.sens_mat[0]),label="+ 1-year Galactic foreground")
ax.loglog(f_curve,np.sqrt(four_year.sens_mat[0]),label="+ 4-year Galactic foreground")
ax.set(xlabel="frequency [Hz]",ylabel=r"TDI A ASD [1/$\\sqrt{\\mathrm{Hz}}$]",title="Sensitivity is part of the likelihood"); ax.legend(); plt.show()'''),
    md(r'''## 3. Inner product, SNR, and likelihood

For independent A and E channels,
\[
(a\mid b)=4\Delta f\,\mathrm{Re}\sum_{X\in\{A,E\},k}\frac{a_{Xk}^*b_{Xk}}{S_X(f_k)},
\quad\rho_{\rm opt}=\sqrt{(h\mid h)},
\quad\log\mathcal L=-\tfrac12(d-h\mid d-h).
\]

These are the same objects as in LVK analysis. What changes is the instrument response, source durations, channels, band, and global model.'''),
    code('''t_obs=90*86400.; simulator=JaxGB(orbits,t_obs=t_obs,t0=0,n=128)
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
axes[1].set(xlabel="frequency [mHz]",ylabel=r"contribution to $\\rho^2$",title="PSD-weighted information by bin"); plt.show()'''),
    md('''Perturb the source frequency and compare optimal SNR with detected/matched SNR. A loud template can still match the data poorly.'''),
    code('''offsets=np.linspace(-7e-7,7e-7,61); detected=[]; logL=[]
for offset in offsets:
    trial=GBObject(f0=np.array([3e-3+offset]),fdot=np.array([1e-17]),A=np.array([2e-22]),ra=np.array([1.]),dec=np.array([.4]),psi=np.array([.3]),iota=np.array([.8]),phi0=np.array([.2]),t_init=0.)
    p=trial.to_jaxgb_array(t0=0); a,e,_=simulator.sum_tdi(p,int(simulator.get_kmin(parameters[:,0])[0]),int(simulator.get_kmin(parameters[:,0])[0])+simulator.n,tdi_generation=2,tdi_combination="AET")
    h=np.stack([np.asarray(a),np.asarray(e)]); rho=np.sqrt(inner(h,h)); detected.append(inner(template,h)/rho); logL.append(-.5*inner(template-h,template-h))
fig,axes=plt.subplots(1,2,figsize=(10,3.3)); axes[0].plot(offsets,detected); axes[0].axhline(optimal_snr,color="k",ls="--")
axes[0].set(xlabel="frequency offset [Hz]",ylabel="detected SNR",title="Match falls away from the signal")
axes[1].plot(offsets,np.array(logL)-np.max(logL)); axes[1].set(xlabel="frequency offset [Hz]",ylabel=r"$\\Delta \\log \\mathcal{L}$",title="Likelihood localises frequency"); plt.show()'''),
    md(r'''## 4. The global-fit problem

The LATW challenge combines a massive-black-hole binary with groups of Galactic binaries. The GLASS demonstration writes the same idea schematically as
\[
d=h_{\rm UCB}+h_{\rm VGB}+h_{\rm MBHB}+n(\eta).
\]
No source is analysed against pristine data: each block sees a residual containing the current estimates of all other source and noise blocks. Source count may itself be unknown, motivating reversible-jump/trans-dimensional methods.'''),
    code('''frequencies=np.array([3e-3,3.00012e-3,3.00025e-3]); true_scales=np.array([1.,.72,.48])
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
ax.set(xlabel="frequency [mHz]",ylabel="TDI A magnitude",title="Overlapping JaxGB sources plus LISA noise"); ax.legend(); plt.show()'''),
    code('''# One sequential pass, then a blocked conditional fit.
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
ax.set(xlabel="blocked sweep",ylabel="amplitude multiplier",title="Source blocks communicate through the residual"); ax.legend(); plt.show()'''),
    md('''## 5. A miniature unknown-source-count challenge

LATW Tutorial 6 uses RJMCMC so the number of Galactic binaries is inferred. Here we enumerate all eight subsets of three candidate templates and use BIC only as a fast classroom proxy—not as a replacement for evidence or RJMCMC.'''),
    code('''model_scores=[]
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
print("Preferred subset:",labels[int(np.argmin(delta))],"(the injected subset is 111)")'''),
    md('''## Extension: animate the global residual'''),
    code('''fig,ax=plt.subplots(figsize=(9,3.3)); line,=ax.plot([],[],lw=.8)
ax.set(xlim=(1e3*common_frequency.min(),1e3*common_frequency.max()),ylim=(0,1.1*np.max(np.abs(data[0]))),xlabel="frequency [mHz]",ylabel="A residual magnitude")
def animate_residual(i):
    residual=data-np.sum(history[i,:,None,None]*templates,axis=0); line.set_data(1e3*common_frequency,np.abs(residual[0])); ax.set_title(f"global residual after sweep {i}"); return (line,)
animation=FuncAnimation(fig,animate_residual,frames=len(history),interval=220); plt.close(fig); display(HTML(animation.to_jshtml()))'''),
    md('''## Optional LISA Data Challenge input and boundary

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
- [LISA Data Challenge files](https://lisa-ldc.in2p3.fr/file).'''),
])
