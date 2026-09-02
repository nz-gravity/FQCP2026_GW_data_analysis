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
#   orphan: true
# ---

# %% [markdown]
# ---
#
# # Worked solutions
#
# Everything above this line is the lab notebook, unchanged. Below are the
# solutions to its exercises, as cells that actually run: execute the notebook
# from the top and every variable they need will already exist.

# %% [markdown]
# ## Code studio: a tiny template bank

# %%
def student_template_bank_scan(offsets):
    peaks = []
    for offset in offsets:
        trial = polarizations(theta_true.at[0].set(float(theta_true[0]) + offset))
        snr_series, _ = matched_filter(h1, trial)
        peaks.append(snr_series.max())
    return np.asarray(peaks)


bank = student_template_bank_scan(np.linspace(-2, 2, 9))
print("peak SNR per template:", np.round(bank, 1))
print(f"best offset: {np.linspace(-2, 2, 9)[np.argmax(bank)]:+.1f} Msun")

# %% [markdown]
# ## Question: how densely must a bank be packed?

# %%
peaks = np.asarray(recovered_peaks)
within = mismatch_offsets[peaks >= 0.97 * peaks.max()]
spacing = within.max() - within.min()
print(f"97% width: {spacing:.2f} Msun")
print(f"templates to cover 20-40 Msun: {int(np.ceil(20 / spacing))}")

# %% [markdown]
# The count is small here only because this is one parameter, at one mass, with a
# Newtonian waveform. An O4 bank covers masses, mass ratio, and aligned spins
# simultaneously and holds a few $10^5$ templates — which is why the matched
# filter has to be an FFT rather than a loop.
