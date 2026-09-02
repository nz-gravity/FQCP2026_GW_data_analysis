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
# ## Question: does subtraction order matter?

# %%
reverse_residual = data.copy()
reverse_amplitudes = np.zeros(3)
for source_index in reverse_order:
    template_i = templates[source_index]
    reverse_amplitudes[source_index] = global_inner(
        template_i, reverse_residual
    ) / global_inner(template_i, template_i)
    reverse_residual -= reverse_amplitudes[source_index] * template_i
print(
    "reverse one-pass residual norm:",
    global_inner(reverse_residual, reverse_residual),
)

# %% [markdown]
# One-pass subtraction fits each source to a residual that still contains the
# other sources' power. That error is baked into the amplitude and never revisited,
# and the next source inherits it — so the answer depends on which source went
# first. The blocked chain revisits every source conditional on the current best
# estimate of all the others, which is why it converges to the same place
# regardless of ordering. This is the entire argument for a global fit over a
# subtract-the-loudest pipeline.
