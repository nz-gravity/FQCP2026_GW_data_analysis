"""Curate the published GW150914 posterior for the GW150914 notebook.

The full GWTC-2.1 data release for this event is a 134 MB HDF5 file. Nobody
should download that in a workshop room, so this script pulls the handful of
columns the notebook overlays and writes them to a small CSV. The notebook
shows the full download too — it just does not depend on it.

Source: GWTC-2.1, https://doi.org/10.5281/zenodo.6513631 (CC BY 4.0).
Analysis ``C01:Mixed``: the catalogue's fiducial samples for this event,
pooled across the IMRPhenomXPHM and SEOBNRv4PHM waveform models.

Run manually when the reference data change; it is not part of the routine
notebook build.
"""

from pathlib import Path
from urllib.request import urlretrieve

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "gw150914_gwtc2p1_posterior.csv"
CACHE = ROOT / "notebooks" / "gw150914_cache" / "IGWN-GWTC2p1-v2-GW150914_095045.h5"
URL = (
    "https://zenodo.org/api/records/6513631/files/"
    "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5/content"
)
ANALYSIS = "C01:Mixed"
COLUMNS = [
    "chirp_mass",
    "chirp_mass_source",
    "mass_1",
    "mass_2",
    "mass_ratio",
    "luminosity_distance",
    "chi_eff",
]


def main():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading 134 MB from Zenodo to {CACHE.relative_to(ROOT)} ...")
        urlretrieve(URL, CACHE)

    with h5py.File(CACHE, "r") as source:
        samples = source[f"{ANALYSIS}/posterior_samples"][()]

    table = np.column_stack([samples[name] for name in COLUMNS])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT,
        table,
        delimiter=",",
        header=",".join(COLUMNS),
        comments="",
        fmt="%.8g",
    )
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(table)} samples, {size_kb:.0f} kB")


if __name__ == "__main__":
    main()
