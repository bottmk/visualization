"""Compare three FFT modes (tilt / output_shift / zero) side by side.

Produces a 3x4 grid (3 modes x 4 theta_i values) of BSDF profiles on v=0 slice
in relative coordinates (u - u_spec).

- tilt:         current default. full hemisphere coverage but has spectral leakage.
- output_shift: no leakage, but back-scatter side missing at large theta_i.
- zero:         normal-incidence approximation. theta_i-independent, same curve.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from bsdf_sim.models.random_rough import RandomRoughSurface
from bsdf_sim.optics.fft_bsdf import compute_bsdf_fft

OUT = Path(__file__).parent / "fft_modes_comparison.png"

WAVELENGTH_UM = 0.55
RQ_UM = 0.02           # 20 nm (intermediate)
LC_UM = 2.0
GRID_SIZE = 512
PIXEL_UM = 0.15        # small enough for output_shift at theta_i up to ~70 deg

THETA_I_LIST = [0.0, 20.0, 45.0, 60.0]
MODES = ["tilt", "output_shift", "zero"]
MODE_COLORS = {"tilt": "tab:blue", "output_shift": "tab:orange", "zero": "tab:green"}


def extract_phi0_slice(u_grid: np.ndarray, bsdf: np.ndarray):
    """Return (u_axis, bsdf_row) at v=0 (index j=0), sorted ascending by u."""
    u_axis = u_grid[:, 0]
    bsdf_row = bsdf[:, 0]
    order = np.argsort(u_axis)
    return u_axis[order], np.maximum(bsdf_row[order], 1e-12)


def main() -> None:
    surface = RandomRoughSurface(
        grid_size=GRID_SIZE, pixel_size_um=PIXEL_UM,
        rq_um=RQ_UM, lc_um=LC_UM, fractal_dim=2.5, seed=42,
    )
    hm = surface.get_height_map()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)

    for ax, mode in zip(axes, MODES):
        for ti_deg in THETA_I_LIST:
            u_grid, _, bsdf = compute_bsdf_fft(
                hm, wavelength_um=WAVELENGTH_UM,
                theta_i_deg=ti_deg, phi_i_deg=0.0,
                fft_mode=mode,
            )
            u_axis, bsdf_row = extract_phi0_slice(u_grid, bsdf)
            u_spec = np.sin(np.deg2rad(ti_deg)) if mode != "zero" else 0.0
            u_rel = u_axis - u_spec
            ax.plot(u_rel, bsdf_row, linewidth=1.2, label=f"theta_i = {ti_deg:.0f} deg")

        ax.set_yscale("log")
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel("u - u_spec  (relative direction cosine)")
        ax.set_title(f"mode = '{mode}'")
        ax.grid(True, which="both", alpha=0.3)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.legend(loc="upper right", fontsize=9)

    axes[0].set_ylabel("BSDF [1/sr]")

    fig.suptitle(
        f"FFT mode comparison (v=0 slice, relative coords)  |  "
        f"RandomRough Rq={RQ_UM*1000:.0f}nm, lc={LC_UM}um, "
        f"N={GRID_SIZE}, dx={PIXEL_UM}um, lambda={WAVELENGTH_UM*1000:.0f}nm, BRDF",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
