---
orphan: true
---

# Instructor run sheet — 120 minute lecture

Instructor material. Deliberately kept out of `_toc.yml`, so it has a URL but
does not appear in the student navigation.

Every beat below links into the built site. Open this page on the podium
machine and drive the lecture from it.

## Format

No laptops in the room. Serve the built site and present it full screen:

```bash
uv run --locked python -m http.server -d _build/html 8000
```

Every figure and animation is already rendered, so nothing samples live in
front of the room. Students get the Colab links at the end and run the labs
themselves.

Exercises become **predictions**. Ask the question, take answers from the room,
then scroll to the executed output. The `*_answers` pages are the reveal — open
them in a second tab, never on screen before the question has been asked.

Before you walk in: open `http://localhost:8000`, zoom to ~150 %, check the
back row can read a code cell, and click one animation to confirm it plays.

## Budget

| Time | Block | Page |
| --- | --- | --- |
| 0:00 | Opening — the inference map | [index](index.md) |
| 0:05 | 1. Bayes from scratch | [01](notebooks/01_bayesian_inference.ipynb) |
| 0:30 | 2. LVK signals and matched filtering | [02](notebooks/02_lvk_signals_injections.ipynb) |
| 0:50 | 3. GW150914, end to end | [03](notebooks/03_lvk_gw150914_bilby.ipynb) |
| 1:10 | Buffer / questions | — |
| 1:15 | 4. Populations and selection | [04](notebooks/04_lvk_population_and_checks.ipynb) |
| 1:25 | 5. LISA: why it is different | [05](notebooks/05_lisa_signals_response_codes.ipynb) |
| 1:40 | 6. The global fit as Gibbs | [06](notebooks/06_lisa_global_fit_gibbs.ipynb) |
| 1:55 | Closing and take-home | [index](index.md) |

Eight modules do not fit in two hours. Modules 07, 08 and 10 are signposted in
the closing block, not walked through. If you are running late, drop block 4
first — it is the shortest live route in the course and the easiest to
summarise in two sentences.

## Opening — 5 min

$$\text{data} + \text{signal model} + \text{noise model}
\to \text{likelihood} \to \text{posterior} \to \text{checks} \to \text{claim}.$$

Say the thing the whole course rests on: the data, the response and the noise
model change from LVK to LISA; the logic in that line does not. Promise you
will point at this map at the start of every block.

## Block 1 — Bayes from scratch (25 min)

[01_bayesian_inference](notebooks/01_bayesian_inference.ipynb) — stop at "End
of the live route".

- <a href="notebooks/01_bayesian_inference.html#the-four-pieces-of-bayes-theorem">The four pieces of Bayes' theorem</a>
  — stress that the model is *chosen*, not given.
- <a href="notebooks/01_bayesian_inference.html#priors-and-prior-predictive-checks">Priors and prior predictive checks</a>
  — draw from the prior, look at the data it implies. Cheapest bug-catcher in
  the course.
- <a href="notebooks/01_bayesian_inference.html#tool-1-the-posterior-on-a-grid">Tool 1: the posterior on a grid</a>
  — exact, and it dies in dimension. That death is why samplers exist.
- <a href="notebooks/01_bayesian_inference.html#tool-2-the-posterior-from-a-sampler">Tool 2: the posterior from a sampler</a>
  — same posterior, no grid.
- **ASK:** *"<a href="notebooks/01_bayesian_inference.html#exercise-1-does-more-data-help">Exercise 1</a>
  — I double the number of data points. What happens to the width of the
  posterior, and why?"* Two answers from the room, then
  [reveal](notebooks/01_bayesian_inference_answers.html).
- <a href="notebooks/01_bayesian_inference.html#posterior-predictive-check">Posterior predictive check</a>,
  then <a href="notebooks/01_bayesian_inference.html#the-check-that-fails">the check that fails</a>.
  Spend real time here: a posterior can be perfectly computed and still answer
  the wrong question.
- <a href="notebooks/01_bayesian_inference.html#the-gravitational-wave-bridge-the-same-likelihood-in-frequency">The gravitational-wave bridge</a>
  and <a href="notebooks/01_bayesian_inference.html#from-residuals-to-the-whittle-likelihood">from residuals to Whittle</a>.
  This is the hinge of the course: everything after it is the same Gaussian
  likelihood with a harder forward model.

Skip in the room:
<a href="notebooks/01_bayesian_inference.html#is-there-a-signal-at-all-evidence-and-the-bayes-factor">evidence and Bayes factors</a>,
exercises 2–4. Name them as take-home.

## Block 2 — LVK signals and matched filtering (20 min)

[02_lvk_signals_injections](notebooks/02_lvk_signals_injections.ipynb) — stop
at "End of the live route".

- <a href="notebooks/02_lvk_signals_injections.html#three-questions-that-must-stay-separate">Three questions that must stay separate</a>
  — is there a signal / what is it / how do we know we are right. Every later
  confusion is one of these three collapsing into another.
- **ANIMATION:** <a href="notebooks/02_lvk_signals_injections.html#animation-why-chirp-mass-is-measured-so-precisely">why chirp mass is measured so precisely</a>.
  Best thirty seconds in the course for a non-specialist audience.
- <a href="notebooks/02_lvk_signals_injections.html#from-source-to-a-detector-network">From source to a detector network</a>.
- <a href="notebooks/02_lvk_signals_injections.html#finding-the-signal-first-matched-filtering">Matched filtering</a>,
  then **ANIMATION:** <a href="notebooks/02_lvk_signals_injections.html#animation-sliding-the-template-through-whitened-data">sliding the template through whitened data</a>.
  Follow immediately with
  <a href="notebooks/02_lvk_signals_injections.html#a-template-only-works-if-it-is-close-enough">a template only works if it is close enough</a>.
- **ASK:** *"<a href="notebooks/02_lvk_signals_injections.html#question-how-densely-must-a-bank-be-packed">how densely must a bank be packed?</a>"*
  Take the qualitative half only: if each extra parameter multiplies the
  template count, what does that do to the cost of a search? Arithmetic is
  take-home.
- <a href="notebooks/02_lvk_signals_injections.html#inject-and-infer-manually">Inject and infer manually</a>,
  then <a href="notebooks/02_lvk_signals_injections.html#put-the-same-likelihood-behind-bilby-s-interface">the same likelihood behind Bilby's interface</a>.
  Land the point: Bilby is an interface, not a new method.

Signpost only:
<a href="notebooks/02_lvk_signals_injections.html#why-a-network-localises-the-sky">sky localisation</a>,
<a href="notebooks/02_lvk_signals_injections.html#a-two-dimensional-posterior-with-a-real-degeneracy">distance–inclination degeneracy</a>.

## Block 3 — GW150914, end to end (20 min)

[03_lvk_gw150914_bilby](notebooks/03_lvk_gw150914_bilby.ipynb). Attribute the
GWOSC 5.2 tutorial out loud —
<a href="notebooks/03_lvk_gw150914_bilby.html#source-and-attribution">source and attribution</a>.

- <a href="notebooks/03_lvk_gw150914_bilby.html#get-the-gw150914-analysis-data">Get the analysis data</a>,
  then <a href="notebooks/03_lvk_gw150914_bilby.html#why-does-gw150914-need-only-four-seconds">why only four seconds</a>
  and <a href="notebooks/03_lvk_gw150914_bilby.html#what-those-analysis-windows-contain-in-the-time-domain">what the windows contain</a>.
- <a href="notebooks/03_lvk_gw150914_bilby.html#estimate-the-psd-from-off-source-data">Estimate the PSD off-source</a>,
  with <a href="notebooks/03_lvk_gw150914_bilby.html#optional-visual-explainer-why-use-a-median">why a median</a>.
- **ASK:** *"<a href="notebooks/03_lvk_gw150914_bilby.html#question">why estimate the PSD away from the signal?</a>"*
  Short, everyone can answer — a good pace-recovery point if you are behind.
- <a href="notebooks/03_lvk_gw150914_bilby.html#define-the-restricted-prior">The restricted prior</a>.
  Say plainly that the non-spinning model and fast sampler settings are
  workshop compromises, not the published analysis.
- <a href="notebooks/03_lvk_gw150914_bilby.html#run-dynesty">Run Dynesty</a>, then
  <a href="notebooks/03_lvk_gw150914_bilby.html#compare-with-the-published-posterior">compare with the published posterior</a>.
- **ASK:** *"do the two posteriors agree — and what should 'agree' even mean:
  overlapping intervals, consistent medians, or matching widths?"*
  (<a href="notebooks/03_lvk_gw150914_bilby.html#id1">the question</a>) Best discussion
  question in the course. Leave 3 minutes for it.

## Buffer — 5 min

Questions. If none come, this is where block 3's overlay discussion runs long
by design.

## Block 4 — Populations and selection (10 min)

[04_lvk_population_and_checks](notebooks/04_lvk_population_and_checks.ipynb) —
live route ends early; genuinely short.

- <a href="notebooks/04_lvk_population_and_checks.html#start-with-the-catalogue-you-detected">Start with the catalogue you detected</a>
  — and it is not the population.
- <a href="notebooks/04_lvk_population_and_checks.html#the-selection-correction-in-one-equation">The selection correction in one equation</a>.
- **ASK:** *"<a href="notebooks/04_lvk_population_and_checks.html#question">the naive estimate is biased in a predictable direction — which
  direction?</a>"* Then
  [reveal](notebooks/04_lvk_population_and_checks_answers.html).
- State the boundary: this is toy population inference, not production
  hierarchical inference.
  <a href="notebooks/04_lvk_population_and_checks.html#extension-events-are-posteriors-not-numbers">Events are posteriors, not numbers</a>
  is take-home.

## Block 5 — LISA: why it is different (15 min)

[05_lisa_signals_response_codes](notebooks/05_lisa_signals_response_codes.ipynb)
— stop at "End of the live route".

- <a href="notebooks/05_lisa_signals_response_codes.html#lisa-s-band-and-source-zoo">Band and source zoo</a>,
  then <a href="notebooks/05_lisa_signals_response_codes.html#why-lisa-parameter-estimation-is-unusually-coupled">why LISA PE is unusually coupled</a>
  — everything is on at once, forever.
- **ANIMATION:** <a href="notebooks/05_lisa_signals_response_codes.html#animation-orbital-motion-becomes-response-modulation">orbital motion becomes response modulation</a>.
  Play it. This is the single idea that separates LISA from LVK in most
  people's heads.
- <a href="notebooks/05_lisa_signals_response_codes.html#sensitivity-and-galactic-confusion">Sensitivity and Galactic confusion</a>.
- <a href="notebooks/05_lisa_signals_response_codes.html#inner-product-snr-and-likelihood">Inner product, SNR, and likelihood</a>
  — point back at the opening map: same likelihood, harder response.
- <a href="notebooks/05_lisa_signals_response_codes.html#analysis-code-map">Analysis-code map</a>
  — Erebor / Gemoo / GLASS / Eryn, one sentence each, so the names are not a
  mystery when they meet them in a talk.

Signpost only:
<a href="notebooks/05_lisa_signals_response_codes.html#extension-from-one-way-links-to-xyz-aet">links and delays into XYZ/AET</a>,
<a href="notebooks/05_lisa_signals_response_codes.html#extension-fisher-forecasts-for-lisa">Fisher forecasts</a>.

## Block 6 — The global fit as Gibbs (15 min)

[06_lisa_global_fit_gibbs](notebooks/06_lisa_global_fit_gibbs.ipynb) — stop at
"End of the live route".

- <a href="notebooks/06_lisa_global_fit_gibbs.html#the-global-fit-a-wheel-of-conditional-analyses">The wheel of conditional analyses</a>.
- <a href="notebooks/06_lisa_global_fit_gibbs.html#gibbs-versus-blocked-metropolis-hastings">Gibbs versus blocked Metropolis-Hastings</a>.
- **ANIMATION:** <a href="notebooks/06_lisa_global_fit_gibbs.html#animation-one-conditional-block-at-a-time">one conditional block at a time</a>.
  Narrate the shared residual being handed on — that hand-off *is* the global
  fit.
- **ASK:** *"<a href="notebooks/06_lisa_global_fit_gibbs.html#question">why can one-pass subtraction depend on the order you remove
  sources in?</a>"* Answers
  first, then [reveal](notebooks/06_lisa_global_fit_gibbs_answers.html).
- Boundary: a fixed catalogue with BIC is not a trans-dimensional global fit.

## Closing — 5 min

Three things you did not walk through, one sentence and one figure each:

- <a href="notebooks/07_lisa_pspline_psd.html#fit-the-penalised-whittle-objective">07 — P-spline PSDs</a>:
  the PSD is estimated *inside* the analysis, with uncertainty.
- <a href="notebooks/08_lisa_wdm_time_frequency.html#wilson-daubechies-meyer-wdm-time-frequency-map-and-likelihood">08 — WDM time-frequency</a>:
  gaps and non-stationarity; a diagonal WDM likelihood is an approximation once
  they induce covariance.
- <a href="notebooks/10_fast_likelihoods.html#the-check-that-decides-whether-you-may-use-them">10 — fast likelihoods</a>:
  heterodyning and relative binning, and the accuracy check that decides
  whether you are allowed to use them.

Then: everything is on the site, every notebook opens in Colab with the
{guilabel}`rocket` button and runs top to bottom on its own. Point at
[the blind data challenge](notebooks/04b_lvk_blind_data_challenge.ipynb) as the
thing to do if they only do one, and at [the glossary](glossary.md) and
[the literature list](notebooks/03_literature.md) for what to read next. Close
on the map you opened with.
