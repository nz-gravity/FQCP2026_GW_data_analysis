"""Build the Colab-first FQCP 2026 gravitational-wave inference course."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks"

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

def write(name, title, cells):
    header = md(f"""# {title}

**FQCP 2026 · Bayesian parameter estimation for gravitational-wave sources**

> Designed for Google Colab. Run top to bottom; **Extension** sections may be skipped live.
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

OUT.mkdir(parents=True, exist_ok=True)
for legacy in ("03_population_inference.ipynb", "04_lisa_global_fit.ipynb"):
    (OUT / legacy).unlink(missing_ok=True)

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

write("00_start_here.ipynb", "Start here: the map from source to posterior", [
    md(r'''## The question behind the whole course

How do source parameters become distributions inferred from noisy detector data?

\[
\theta_{\rm source}\rightarrow(h_+,h_\times)\rightarrow h_I
\rightarrow(d_I-h_I)\rightarrow p(\theta\mid d).
\]

The first worksheet compressed the middle of this chain. This version makes it explicit: a waveform is not yet detector data, and a global LISA fit is not merely two sinusoids.

## Chapters and two-hour live route

| Chapter | Live use | Main idea |
| --- | ---: | --- |
| 01 Bayes, PSD, Whittle | 25 min | estimate and use coloured noise |
| 02 CBC physics + rippleGW | 25 min | parameters make two polarisations |
| 03 LVK response + Bilby | 30 min | projection, timing, polarisation, localisation |
| 04 Population inference | self-study | selection changes a catalogue |
| 05 LISA response + global fit | 35 min | moving constellation, TDI, overlapping sources |

Keep five minutes for the opening and ten for questions/transitions.'''),
    STANDARD_SETUP,
    md(r'''## Vocabulary and reading strategy

| Term | Meaning |
| --- | --- |
| polarisation | the tensor components $h_+$ and $h_\times$ |
| response | how sky/orientation and detector geometry map polarisations to data |
| PSD / ASD | one-sided noise power / amplitude spectral density; ASD is $\sqrt{S_n}$ |
| likelihood | compatibility of parameters with data and a noise model |
| posterior | prior updated by the likelihood |
| TDI | delayed combinations of LISA laser links that suppress laser-frequency noise |

Each chapter has a physical question, equations, inspectable code, and a **boundary** stating what is simplified. The notebooks are independent. Package installs are visible and pinned. Before teaching, execute everything in a fresh Colab runtime and do not make the live LISA lesson depend on an authenticated data download.'''),
])

write("01_bayes_and_whittle.ipynb", "1. The PSD is part of the likelihood", [
    md(r'''## Learning goals

Distinguish PSD from ASD, estimate a PSD with Welch averaging, whiten a residual, and explain the Whittle likelihood.

\[
\log p(d\mid\theta,S_n)=-\tfrac12(d-h\mid d-h)-\tfrac12\sum_k\log S_n(f_k)+C,
\quad (a\mid b)=4\mathrm{Re}\sum_k\frac{\tilde a_k\tilde b_k^*}{S_n(f_k)}\Delta f.
\]

The PSD is not decorative: it says which residuals are surprising. The normalisation is constant only while the PSD is fixed.'''),
    STANDARD_SETUP,
    code('''from scipy.signal import welch
sample_rate, duration = 512, 64
time = np.arange(0, duration, 1/sample_rate)
frequency = np.fft.rfftfreq(time.size, 1/sample_rate)
target_shape = 1 + (30/np.maximum(frequency, 1))**4 + (frequency/180)**2
white = rng.normal(size=time.size)
noise = np.fft.irfft(np.fft.rfft(white)*np.sqrt(target_shape), n=time.size)
f_welch, psd_hat = welch(noise, fs=sample_rate, nperseg=2048, average="median")
target_asd = np.sqrt(np.interp(f_welch, frequency, target_shape))
target_asd *= np.median(np.sqrt(psd_hat[20:])/target_asd[20:])
fig, ax = plt.subplots(figsize=(8,3.4))
ax.loglog(f_welch[1:], np.sqrt(psd_hat[1:]), label="Welch ASD estimate")
ax.loglog(f_welch[1:], target_asd[1:], "--", label="injected shape (scaled)")
ax.set(xlabel="frequency [Hz]", ylabel=r"ASD $\\sqrt{S_n(f)}$ [toy units]",
       title="PSD estimation averages noisy periodograms"); ax.legend(); plt.show()'''),
    code('''analysis_duration = 8; n = analysis_duration*sample_rate
t = time[:n]; d = noise[:n].copy()
true_amplitude, signal_frequency = 2.2, 45.0
template = np.sin(2*np.pi*signal_frequency*t + .3); d += true_amplitude*template
f = np.fft.rfftfreq(n, 1/sample_rate); df = f[1]-f[0]
Sn = np.interp(f, f_welch, psd_hat); Sn[0] = Sn[1]
d_f = np.fft.rfft(d)
def log_likelihood(amplitude):
    r = d_f-np.fft.rfft(amplitude*template)
    return -2*df*np.sum(np.abs(r)**2/Sn)
grid=np.linspace(0,4,500); logp=np.array([log_likelihood(a) for a in grid])
p=np.exp(logp-logp.max()); p/=np.trapezoid(p,grid); best=grid[np.argmax(p)]
r=d_f-np.fft.rfft(best*template)
fig,axes=plt.subplots(1,2,figsize=(11,3.4))
axes[0].plot(grid,p); axes[0].axvline(true_amplitude,color="k",ls="--")
axes[0].set(xlabel="amplitude",ylabel="posterior density",title="PSD-weighted inference")
axes[1].semilogy(f[1:],np.abs(r[1:]),alpha=.55,label="raw residual")
axes[1].semilogy(f[1:],np.abs(r[1:])/np.sqrt(Sn[1:]),label="whitened residual")
axes[1].set(xlabel="frequency [Hz]",ylabel="magnitude",title="Whitening divides by ASD"); axes[1].legend(); plt.show()
print(f"Injected {true_amplitude:.2f}; MAP {best:.2f}")'''),
    md('''## Exercise and boundary

Replace `Sn` by a constant, then underestimate it near 45 Hz. Which bins dominate and why does the posterior become overconfident?

This example assumes stationary Gaussian noise, independent Fourier bins, a fixed PSD, and no glitches or calibration uncertainty. Those are modelling assumptions, not guarantees supplied by a Fourier transform.'''),
])

write("02_lvk_cbc_with_ripple.ipynb", "2. CBC parameters and the two polarisations", [
    md(r'''## Source parameters

| Group | Parameters | Main waveform effect |
| --- | --- | --- |
| masses | $m_1,m_2$ or $\mathcal M,q$ | phase evolution and merger frequency |
| spins | $\vec\chi_1,\vec\chi_2$ | phasing, precession, merger |
| matter/orbit | tides, eccentricity | extra phase and harmonics |
| scale | luminosity distance $D_L$ | amplitude $\propto D_L^{-1}$ |
| orientation | inclination $\iota$, polarisation $\psi$ | polarisation content and sky-basis rotation |
| location/time | right ascension $\alpha$, declination $\delta$, $t_c$ | response and arrival delays |
| phase | coalescence phase $\phi_c$ | waveform phase |

Cosmological analyses use detector-frame masses: $m_{\rm det}=(1+z)m_{\rm source}$. This chapter produces radiation before detector projection; Chapter 03 adds the sky and network.'''),
    code('''import os, sys, subprocess, importlib.util
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
if importlib.util.find_spec("ripplegw") is None:
    if IN_COLAB: subprocess.check_call([sys.executable,"-m","pip","install","-q","rippleGW==0.2.1"])
    else: raise ImportError("Install rippleGW==0.2.1, or run in Colab.")'''),
    code('''import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from jax import config, jit
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
rng=np.random.default_rng(20260817); plt.style.use("seaborn-v0_8-whitegrid"); plt.rcParams["animation.html"]="jshtml"
frequency=jnp.linspace(20.,512.,985); df=float(frequency[1]-frequency[0]); f_ref=20.
def parameters(m1=36.,m2=29.,chi1=.1,chi2=-.1,distance=400.,tc=0.,phase=0.,inclination=.4):
    chirp_mass,eta=ms_to_Mc_eta(jnp.array([m1,m2]))
    return jnp.array([chirp_mass,eta,chi1,chi2,distance,tc,phase,inclination])
@jit
def waveform(theta): return gen_IMRPhenomD_hphc(frequency,theta,f_ref)
theta_true=parameters(); hp_true,hx_true=waveform(theta_true)
print("chirp mass",round(float(theta_true[0]),3),"solar masses; eta",round(float(theta_true[1]),4))'''),
    code('''fig,axes=plt.subplots(1,2,figsize=(11,3.4))
axes[0].loglog(np.asarray(frequency),np.abs(np.asarray(hp_true)),label=r"$h_+$")
axes[0].loglog(np.asarray(frequency),np.abs(np.asarray(hx_true)),label=r"$h_\times$")
axes[0].set(xlabel="frequency [Hz]",ylabel="strain / Hz",title="Two tensor polarisations"); axes[0].legend()
axes[1].plot(np.asarray(frequency),np.unwrap(np.angle(np.asarray(hp_true))))
axes[1].set(xlabel="frequency [Hz]",ylabel="unwrapped phase [rad]",title="Mass information lives strongly in phase"); plt.show()'''),
    md(r'''For a non-precessing circular binary at leading order,
$h_+\propto(1+\cos^2\iota)/(2D_L)$ and $h_\times\propto\cos\iota/D_L$.
Inclination describes the binary relative to the line of sight. Polarisation angle $\psi$ rotates the two axes on the sky; it is a different parameter.'''),
    code('''fig,ax=plt.subplots(figsize=(8,3.4))
for iota in np.linspace(0,np.pi/2,5):
    hp,_=waveform(parameters(inclination=float(iota)))
    ax.loglog(np.asarray(frequency),np.abs(np.asarray(hp)),label=fr"$\\iota={iota:.2f}$")
ax.set(xlabel="frequency [Hz]",ylabel=r"$|h_+|$",title="Inclination changes polarisation amplitude"); ax.legend(ncol=2); plt.show()'''),
    md('''## Animation: chirp mass changes amplitude and phase'''),
    code('''frames=np.linspace(18,42,30); fig,(aa,ap)=plt.subplots(1,2,figsize=(11,3.3)); la,=aa.loglog([],[]); lp,=ap.plot([],[])
aa.set(xlim=(20,512),ylim=(2e-25,4e-22),xlabel="frequency [Hz]",ylabel=r"$|h_+|$")
ap.set(xlim=(20,512),ylim=(-250,10),xlabel="frequency [Hz]",ylabel="relative phase [rad]")
def update_mass(i):
    hp,_=waveform(theta_true.at[0].set(frames[i])); h=np.asarray(hp); phase=np.unwrap(np.angle(h)); phase-=phase[0]
    la.set_data(np.asarray(frequency),np.abs(h)); lp.set_data(np.asarray(frequency),phase); fig.suptitle(f"chirp mass = {frames[i]:.1f} solar masses"); return la,lp
animation=FuncAnimation(fig,update_mass,frames=len(frames),interval=130); plt.close(fig); display(HTML(animation.to_jshtml()))'''),
    code('''f=np.asarray(frequency); psd=(2.2e-23)**2*((35/f)**4+1+(f/260)**2)
noise=np.sqrt(psd/(4*df))*(rng.normal(size=f.size)+1j*rng.normal(size=f.size)); data=np.asarray(hp_true)+noise
def log_likelihood(model): return -2*df*np.sum(np.abs(data-np.asarray(model))**2/psd)
mass_grid=np.linspace(float(theta_true[0])-5,float(theta_true[0])+5,220)
logp=np.array([log_likelihood(waveform(theta_true.at[0].set(m))[0]) for m in mass_grid])
p=np.exp(logp-logp.max()); p/=np.trapezoid(p,mass_grid)
fig,ax=plt.subplots(figsize=(7,3)); ax.plot(mass_grid,p); ax.axvline(float(theta_true[0]),color="k",ls="--",label="injection")
ax.set(xlabel="detector-frame chirp mass [solar masses]",ylabel="posterior density",title="rippleGW in our Whittle likelihood"); ax.legend(); plt.show()'''),
    md('''## Boundary

rippleGW supplied real IMRPhenomD polarisations. The inference still used one polarisation, an analytic PSD, and one free parameter. Next we expose the transformation Bilby normally performs.

- [rippleGW documentation](https://ripplegw.readthedocs.io/)
- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)'''),
])

write("03_lvk_response_and_localisation.ipynb", "3. From polarisations to an LVK detector network", [
    md(r'''## Response and localisation

\[
\tilde h_I(f)=[F^I_+\tilde h_+(f)+F^I_\times\tilde h_\times(f)]e^{-2\pi if\Delta t_I}.
\]

Sky position is shared, but each site has different antenna factors, arrival delay, PSD, and calibration. The coherent network log likelihood sums detector contributions.'''),
    code('''import os, sys, subprocess, importlib.util
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
if importlib.util.find_spec("bilby") is None:
    if IN_COLAB: subprocess.check_call([sys.executable,"-m","pip","install","-q","bilby==2.8.0"])
    else: raise ImportError("Install bilby==2.8.0, or run in Colab.")'''),
    STANDARD_SETUP,
    code('''import bilby
gps_time=1126259462.4; ra_true,dec_true,psi_true=1.2,-.4,.7
ifos=bilby.gw.detector.InterferometerList(["H1","L1","V1"])
print("Detector   F+       Fx      geocentre delay [ms]")
for ifo in ifos:
    fp=ifo.antenna_response(ra_true,dec_true,gps_time,psi_true,"plus")
    fx=ifo.antenna_response(ra_true,dec_true,gps_time,psi_true,"cross")
    dt=ifo.time_delay_from_geocenter(ra_true,dec_true,gps_time)
    print(f"{ifo.name:>5s}   {fp:+.3f}   {fx:+.3f}       {1e3*dt:+.2f}")'''),
    md('''Changing $\\psi$ rotates the plus/cross sky basis; it does not move the source. Differently oriented detectors help resolve the two components.'''),
    code('''psi=np.linspace(0,np.pi,240); fig,ax=plt.subplots(figsize=(8,3.4))
for ifo in ifos:
    fp=[ifo.antenna_response(ra_true,dec_true,gps_time,p,"plus") for p in psi]
    fx=[ifo.antenna_response(ra_true,dec_true,gps_time,p,"cross") for p in psi]
    ax.plot(psi,fp,label=f"{ifo.name} F+"); ax.plot(psi,fx,"--",alpha=.8,label=f"{ifo.name} Fx")
ax.set(xlabel=r"polarisation angle $\\psi$ [rad]",ylabel="antenna factor",title="Same sky position, rotated basis"); ax.legend(ncol=3,fontsize=8); plt.show()'''),
    md('''## Timing-only localisation

Two sites constrain a delay ring; a third reduces it. A real posterior also uses coherent phase, amplitudes, polarisation, waveform uncertainty, Earth rotation, and priors.'''),
    code('''ra=np.linspace(-np.pi,np.pi,91); dec=np.linspace(-np.pi/2,np.pi/2,46); RA,DEC=np.meshgrid(ra,dec)
delays={ifo.name:np.array([[ifo.time_delay_from_geocenter(r,d,gps_time) for r in ra] for d in dec]) for ifo in ifos}
obs={ifo.name:ifo.time_delay_from_geocenter(ra_true,dec_true,gps_time) for ifo in ifos}; sigma_t=3e-4
def timing_ll(names):
    ref=names[0]; value=np.zeros_like(RA)
    for name in names[1:]: value-=.5*((delays[name]-delays[ref]-(obs[name]-obs[ref]))/sigma_t)**2
    return value
fig,axes=plt.subplots(1,2,figsize=(12,4),subplot_kw={"projection":"mollweide"})
for ax,names,title in zip(axes,[["H1","L1"],["H1","L1","V1"]],["two sites: timing ring","three sites: smaller regions"]):
    ll=timing_ll(names); density=np.exp(ll-ll.max()); ax.contourf(RA,DEC,density,levels=np.linspace(.05,1,15),cmap="magma")
    ax.plot(ra_true,dec_true,"c*",ms=10); ax.set_title(title); ax.grid(True)
plt.show()'''),
    code('''fig,ax=plt.subplots(figsize=(8,3.4)); f=np.linspace(10,1024,2000)
for ifo in ifos: ax.loglog(f,ifo.power_spectral_density.get_amplitude_spectral_density_array(f),label=ifo.name)
ax.set(xlabel="frequency [Hz]",ylabel=r"ASD [1/$\\sqrt{\\mathrm{Hz}}$]",title="Detector-specific design curves in Bilby",ylim=(1e-24,1e-20)); ax.legend(); plt.show()'''),
    md('''## “Yay Bilby”: project both polarisations in one call

The narrow-band polarisations below stand in for rippleGW output. `get_detector_response` applies antenna factors and timing. A CBC likelihood invokes this machinery repeatedly.'''),
    code('''for ifo in ifos: ifo.set_strain_data_from_zero_noise(sampling_frequency=1024,duration=4,start_time=gps_time-2)
f=ifos[0].frequency_array; hp=np.exp(-.5*((f-100)/20)**2).astype(complex); hx=-.7j*hp
parameters=dict(ra=ra_true,dec=dec_true,psi=psi_true,geocent_time=gps_time)
fig,(aa,ap)=plt.subplots(1,2,figsize=(11,3.4))
for ifo in ifos:
    h=ifo.get_detector_response({"plus":hp,"cross":hx},parameters,frequencies=f); keep=np.abs(hp)>1e-3
    aa.plot(f[keep],np.abs(h[keep]),label=ifo.name); ap.plot(f[keep],np.unwrap(np.angle(h[keep])),label=ifo.name)
aa.set(xlabel="frequency [Hz]",ylabel="projected amplitude",title="Antenna amplitudes")
ap.set(xlabel="frequency [Hz]",ylabel="phase [rad]",title="Arrival-time phases"); aa.legend(); ap.legend(); plt.show()'''),
    code('''print("""waveform_generator = bilby.gw.WaveformGenerator(...)
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator)
result = bilby.run_sampler(likelihood, priors, ...)

Bilby handles the response and coherent likelihood for us. Yay Bilby!""")'''),
    md('''## Boundary

The map is timing-only, and the PSDs are design curves rather than event estimates. Production work samples nuisance parameters and tests waveform, PSD, and calibration choices.

- [Bilby detector API](https://bilby-dev.github.io/bilby/api/bilby.gw.detector.interferometer.Interferometer.html)
- [Bilby CBC tutorial](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html)'''),
])

write("04_population_inference.ipynb", "4. From individual binaries to a population", [
    md(r'''## Goal

The detected catalogue is not the astrophysical population:
\[
p(\Lambda\mid\{d_i\},\mathrm{det})\propto p(\Lambda)\prod_i
\frac{\int p(d_i\mid\theta)p(\theta\mid\Lambda)d\theta}{\alpha(\Lambda)}.
\]
For clarity, masses are exact and only a population mean is inferred.'''), STANDARD_SETUP,
    code('''from scipy.stats import norm
true_mean,true_width=28.,5.; all_masses=rng.normal(true_mean,true_width,5000); all_masses=all_masses[(all_masses>8)&(all_masses<55)]
def p_detect(mass): return 1/(1+np.exp(-(mass-22)/3.5))
detected=all_masses[rng.random(all_masses.size)<p_detect(all_masses)][:30]
fig,axes=plt.subplots(1,2,figsize=(10,3.2)); grid=np.linspace(8,55)
axes[0].plot(grid,p_detect(grid)); axes[0].set(xlabel="mass",ylabel="detection probability",title="Toy selection")
axes[1].hist(all_masses,bins=35,density=True,histtype="step",lw=2,label="underlying"); axes[1].hist(detected,bins=12,density=True,alpha=.55,label="detected")
axes[1].set(xlabel="chirp mass [toy units]",ylabel="density",title="Observed is not underlying"); axes[1].legend(); plt.show()'''),
    code('''means=np.linspace(15,40,350); x=np.linspace(8,55,800); naive=[]; corrected=[]
for mean in means:
    event=norm.logpdf(detected,mean,true_width).sum(); alpha=np.trapezoid(norm.pdf(x,mean,true_width)*p_detect(x),x)
    naive.append(event); corrected.append(event-len(detected)*np.log(alpha))
def normalise(logp): p=np.exp(logp-np.max(logp)); return p/np.trapezoid(p,means)
fig,ax=plt.subplots(figsize=(7,3.2)); ax.plot(means,normalise(np.array(naive)),label="ignores selection"); ax.plot(means,normalise(np.array(corrected)),label="selection-aware")
ax.axvline(true_mean,color="k",ls="--",label="injection"); ax.set(xlabel="population mean",ylabel="posterior density",title="Selection changes the answer"); ax.legend(); plt.show()'''),
    md('''## Boundary

Real analyses reuse uncertain event likelihoods/posteriors, infer several hyperparameters, estimate selection with injections, and may include the rate. Every earlier choice—PSD, waveform, response—affects the event posteriors being combined.'''),
])

write("05_lisa_response_and_global_fit.ipynb", "5. LISA response and the global-fit problem", [
    md(r'''## Why LISA is different

| Observatory | Approximate band | Response picture |
| --- | --- | --- |
| LVK | tens of Hz to kHz | short signals in separated, nearly rigid detectors |
| LISA | roughly $10^{-4}$ to $10^{-1}$ Hz | orbital modulation, finite arms, delayed links, TDI |

LISA observes long-lived Galactic binaries, massive-black-hole binaries, EMRIs, stellar-origin binaries, and stochastic signals. Many overlap in time and frequency, while instrument noise is also uncertain.'''),
    code('''import os, sys, subprocess, importlib.util
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
if importlib.util.find_spec("jaxgb") is None:
    if IN_COLAB: subprocess.check_call([sys.executable,"-m","pip","install","-q","jaxgb==0.2.1","astropy==7.2.0"])
    else: raise ImportError("Install jaxgb==0.2.1 and astropy==7.2.0, or run in Colab.")'''),
    code('''import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
from jax import config
config.update("jax_enable_x64", True)
from lisaorbits import EqualArmlengthOrbits
from jaxgb.jaxgb import JaxGB
from jaxgb.params import GBObject
rng=np.random.default_rng(20260817); plt.style.use("seaborn-v0_8-whitegrid"); plt.rcParams["animation.html"]="jshtml"
year=365.25*86400; AU=149597870700.; orbits=EqualArmlengthOrbits(); times=np.linspace(0,year,240)
positions=np.asarray(orbits.compute_position(times,[1,2,3])); fig,ax=plt.subplots(figsize=(5.5,5.5))
for i,label in enumerate(["SC 1","SC 2","SC 3"]): ax.plot(positions[:,i,0]/AU,positions[:,i,1]/AU,label=label)
ax.plot(0,0,"o",color="gold",mec="k",label="Sun"); ax.set(xlabel="heliocentric x [AU]",ylabel="heliocentric y [AU]",title="Equal-arm LISA orbit model",aspect="equal"); ax.legend(); plt.show()'''),
    md(r'''The triangle orbits and cartwheels. A wave reaches six one-way laser links at different retarded times. Time-delay interferometry combines delayed links to suppress laser-frequency noise. Long observations therefore have orbital modulation and finite-arm transfer features, not three constant antenna factors.

## A real JaxGB response

A Galactic binary uses $(f_0,\dot f,A,\alpha,\delta,\psi,\iota,\phi_0)$. JaxGB evaluates the moving-constellation link/TDI response.'''),
    code('''t_obs=90*86400; response=JaxGB(orbits,t_obs=t_obs,t0=0,n=128)
binary=GBObject(f0=np.array([3e-3]),fdot=np.array([1e-17]),A=np.array([2e-22]),ra=np.array([1.]),dec=np.array([.4]),psi=np.array([.3]),iota=np.array([.8]),phi0=np.array([.2]),t_init=0.)
pars=binary.to_jaxgb_array(t0=0); A,E,T=response.get_tdi(pars,tdi_generation=2,tdi_combination="AET"); freq=np.asarray(response.get_frequency_grid(response.get_kmin(pars[:,0])))[0]
fig,axes=plt.subplots(1,3,figsize=(12,3),sharex=True)
for ax,channel,name in zip(axes,[A,E,T],["A","E","T"]): ax.plot(1e3*freq,np.abs(np.asarray(channel)[0])); ax.set_title(f"TDI {name}"); ax.set_xlabel("frequency [mHz]")
axes[0].set_ylabel("response magnitude"); fig.suptitle("One binary, second-generation TDI"); plt.show()'''),
    md(r'''## The global-fit model

The LISA Data Challenge demonstration used
\[
d=h_{\rm UCB}+h_{\rm VGB}+h_{\rm MBHB}+n(\eta),
\]
updating source/noise blocks against residuals containing current estimates of all other blocks. GLASS simultaneously handled thousands of overlapping sources and unknown noise.

Our mini challenge has three nearby JaxGB sources. We fix their nonlinear parameters/noise and infer only amplitude multipliers: small enough to compare one-pass subtraction, joint fitting, and blocked updates.'''),
    code('''frequencies=np.array([3e-3,3.00012e-3,3.00025e-3]); true_scales=np.array([1.,.72,.48])
catalogue=GBObject(f0=frequencies,fdot=np.array([1e-17,.5e-17,1.5e-17]),A=np.full(3,2e-22),ra=np.array([1.,1.4,2.]),dec=np.array([.4,-.2,.7]),psi=np.array([.3,.8,1.1]),iota=np.array([.8,1.2,.5]),phi0=np.array([.2,1.5,2.4]),t_init=0.)
p_all=catalogue.to_jaxgb_array(t0=0); kmins=np.asarray(response.get_kmin(p_all[:,0])); kmin=int(np.min(kmins)); kmax=int(np.max(kmins)+response.n)
templates=[]
for row in np.asarray(p_all):
    a,_,_=response.sum_tdi(row[None,:],kmin,kmax,tdi_generation=2,tdi_combination="AET"); templates.append(np.asarray(a))
templates=np.asarray(templates); model=np.sum(true_scales[:,None]*templates,axis=0); noise_sigma=.08*np.max(np.abs(model))
data=model+noise_sigma*(rng.normal(size=model.size)+1j*rng.normal(size=model.size)); common_f=np.arange(kmin,kmax)/t_obs
fig,ax=plt.subplots(figsize=(9,3.4)); ax.plot(1e3*common_f,np.abs(data),color="k",lw=.8,label="data")
for i,tpl in enumerate(templates): ax.plot(1e3*common_f,np.abs(true_scales[i]*tpl),label=f"source {i+1}")
ax.set(xlabel="frequency [mHz]",ylabel="TDI A magnitude",title="Three overlapping JaxGB responses"); ax.legend(); plt.show()'''),
    code('''sequential=np.zeros(3); residual=data.copy()
for i,tpl in enumerate(templates): sequential[i]=np.real(np.vdot(tpl,residual)/np.vdot(tpl,tpl)); residual-=sequential[i]*tpl
X=templates.T; joint=np.linalg.lstsq(np.vstack([X.real,X.imag]),np.r_[data.real,data.imag],rcond=None)[0]
blocked=np.zeros(3); history=[blocked.copy()]
for iteration in range(12):
    for i,tpl in enumerate(templates):
        effective=data-np.sum(blocked[:,None]*templates,axis=0)+blocked[i]*tpl
        blocked[i]=np.real(np.vdot(tpl,effective)/np.vdot(tpl,tpl))
    history.append(blocked.copy())
history=np.asarray(history)
print("true      ",np.round(true_scales,3)); print("one pass  ",np.round(sequential,3)); print("joint     ",np.round(joint,3)); print("blocked   ",np.round(blocked,3))
fig,ax=plt.subplots(figsize=(8,3.4))
for i in range(3): ax.plot(history[:,i],"o-",label=f"source {i+1}"); ax.axhline(true_scales[i],color=f"C{i}",ls="--",alpha=.5)
ax.set(xlabel="blocked sweep",ylabel="amplitude multiplier",title="Blocks communicate through the global residual"); ax.legend(); plt.show()'''),
    md('''## Extension: animate the residual'''),
    code('''fig,ax=plt.subplots(figsize=(9,3.3)); line,=ax.plot([],[],lw=.8)
ax.set(xlim=(1e3*common_f.min(),1e3*common_f.max()),ylim=(0,1.1*np.max(np.abs(data))),xlabel="frequency [mHz]",ylabel="residual magnitude")
def update(i):
    r=data-np.sum(history[i,:,None]*templates,axis=0); line.set_data(1e3*common_f,np.abs(r)); ax.set_title(f"global residual after sweep {i}"); return (line,)
animation=FuncAnimation(fig,update,frames=len(history),interval=220); plt.close(fig); display(HTML(animation.to_jshtml()))'''),
    md('''## Optional LISA Data Challenge input

The LDC file portal redirects unauthenticated visitors to login. Do not make the live class depend on it; authenticated students may upload a selected file into Colab afterward.'''),
    code('''RUN_LDC_UPLOAD=False
if RUN_LDC_UPLOAD:
    if not IN_COLAB: raise RuntimeError("Upload helper is for Colab.")
    from google.colab import files
    uploaded=files.upload(); print("Uploaded:",list(uploaded))
else: print("Optional authenticated portal: https://lisa-ldc.in2p3.fr/file")'''),
    md('''## Boundary

This uses a real orbit/TDI response but only three Galactic binaries and one fitted TDI channel. Frequencies, phases, sky positions, source count, and noise were fixed. Research global fits need probabilistic sampling, birth/death moves, multiple TDI channels and source classes, inferred noise, and convergence/coverage validation.

- [GLASS global analysis](https://arxiv.org/abs/2301.03673)
- [LISA Data Challenge files](https://lisa-ldc.in2p3.fr/file)
- [JaxGB](https://pypi.org/project/jaxgb/)
- [LISA GW Response](https://lisa-simulation.pages.in2p3.fr/gw-response/)'''),
])
