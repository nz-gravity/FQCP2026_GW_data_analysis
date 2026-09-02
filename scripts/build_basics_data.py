"""Build the mystery dataset for the Part 1 basics notebook.

Students load ``assets/basics_mystery_data.csv`` over the network and never see
this file during the workshop.  That is the point of notebook 01: fit a model,
believe its posterior, watch a Bayes factor call it a confident detection, and
only then find out that the model was the wrong shape all along.

**Instructors:** the generating process is an exponential rise,
``h(t) = A (exp(t / tau) - 1)`` with ``A = 1`` and ``tau = 3``, observed at 100
evenly spaced times on ``[0, 10]`` with white Gaussian noise of ``sigma = 3``.
The notebook tells the students ``sigma``; it tells them nothing else.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "basics_mystery_data.csv"
SEED = 20260817
SIGMA = 3.0
AMPLITUDE = 1.0
TAU = 3.0


def main():
    rng = np.random.default_rng(SEED)
    time = np.linspace(0, 10, 100)
    truth = AMPLITUDE * (np.exp(time / TAU) - 1.0)
    observation = truth + rng.normal(0, SIGMA, time.size)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT,
        np.column_stack([time, observation]),
        delimiter=",",
        header="time,observation",
        comments="",
        fmt="%.10g",
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({time.size} rows)")


if __name__ == "__main__":
    main()
