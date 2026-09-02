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
# ## Question: how local is a gap?

# %%
missing_sample_fraction = 1.0 - available.mean()
gap_columns = (pixel_time >= GAP_DAYS[0]) & (pixel_time <= GAP_DAYS[1])
print(f"missing time samples:     {missing_sample_fraction:.3f}")
print(f"WDM columns over the gap: {gap_columns.mean():.3f}")

# %% [markdown]
# The gap is local in time and local in WDM columns, but **not** local in
# frequency: convolving with the sharp-edged window spreads its power across
# essentially the whole band. That is the entire argument for analysing gapped
# data in a time-frequency basis rather than a pure Fourier one — and also the
# reason a diagonal WDM likelihood is only an approximation near the edges, where
# neighbouring pixels become correlated.

# %% [markdown]
# ## Question: order dependence, again

# %%
reverse_residual = wdm_global_data.copy()
reverse_amplitudes = np.zeros(3)
for source_index in reverse_order:
    template = wdm_global_templates[source_index]
    reverse_amplitudes[source_index] = wdm_global_inner(
        template, reverse_residual
    ) / wdm_global_inner(template, template)
    reverse_residual -= reverse_amplitudes[source_index] * template

joint_residual = wdm_global_data - np.sum(
    wdm_global_joint[:, None, None] * wdm_global_templates, axis=0
)
print(
    "forward one-pass norm:",
    wdm_global_inner(wdm_one_pass_residual, wdm_one_pass_residual),
)
print("reverse one-pass norm:", wdm_global_inner(reverse_residual, reverse_residual))
print("joint-fit norm:       ", wdm_global_inner(joint_residual, joint_residual))

# %% [markdown]
# Both one-pass orderings lose to the joint fit, and they lose by different
# amounts. Moving to a WDM basis localised the *gap*; it did nothing about
# source-model error, because overlapping sources overlap in time-frequency too.
# Choosing a better basis and choosing a better estimator are independent
# decisions, and you need both.
