"""Show the same BSDF curve with 5 different secondary x-axis units.

2x3 panel. Primary x-axis: scattering angle theta_s [deg] (log scale).
Secondary x-axis varies by panel:
  (1) theta_s (no secondary)
  (2) lambda_scale: structure scale Lambda = lambda / sin(theta_s) [um]  (default)
  (3) u: direction cosine u = sin(theta_s)
  (4) f: spatial frequency f = sin(theta_s) / lambda [1/um]
  (5) k_x: transverse wavenumber k_x = 2 pi sin(theta_s) / lambda [rad/um]
  (6) theta_s again (axis-unit reference table as text)

Purpose: visualize the wavevector-family analogy for direction-cosine space.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from bsdf_sim.models.spherical_array import SphericalArraySurface
from bsdf_sim.optics.fft_bsdf import compute_bsdf_fft
from bsdf_sim.visualization.constants import BSDF_LOG_FLOOR_DEFAULT
from bsdf_sim.visualization.profile_extract import slice_phi0
from bsdf_sim.visualization.secondary_axis import (
    AXIS_UNITS,
    add_secondary_xaxis_mpl,
)

OUT = Path(__file__).parent / "secondary_axis_units.png"

WAVELENGTH_UM = 0.55
GRID_SIZE = 1024
PIXEL_UM = 0.15
N1 = 1.0
N2 = 1.5
THETA_I_DEG = 0.0

UNITS_TO_SHOW = ["theta_s", "lambda_scale", "u", "f", "k_x"]


def main() -> None:
    print("Building SphericalArraySurface (R=5um, P=10um, Random)...")
    model = SphericalArraySurface(
        radius_um=5.0, pitch_um=10.0, base_height_um=0.0,
        placement="Random", overlap_mode="Maximum",
        grid_size=GRID_SIZE, pixel_size_um=PIXEL_UM, seed=42,
    )
    hm = model.get_height_map()

    print("Computing BSDF (FFT, theta_i=0)...")
    u_grid, v_grid, bsdf = compute_bsdf_fft(
        hm, wavelength_um=WAVELENGTH_UM,
        theta_i_deg=THETA_I_DEG, phi_i_deg=0.0,
        n1=N1, n2=N2, is_btdf=False,
    )
    u_axis, bsdf_row = slice_phi0(
        u_grid, v_grid, bsdf, mode="positive", floor=BSDF_LOG_FLOOR_DEFAULT,
    )
    theta_s = np.rad2deg(np.arcsin(np.clip(u_axis, 0, 1)))
    # log X plotting: exclude theta_s <= 0.1 deg
    mask = theta_s > 0.1
    theta_s = theta_s[mask]
    bsdf_row = bsdf_row[mask]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    axes_flat = axes.flatten()

    # ── Panels 0..4: each secondary unit ────────────────────────────────
    for i, unit in enumerate(UNITS_TO_SHOW):
        ax = axes_flat[i]
        spec = AXIS_UNITS[unit]
        ax.plot(theta_s, bsdf_row, color="#1f77b4", linewidth=1.4, label="BSDF")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(0.1, 90)
        ax.set_ylim(1e-4, 1e5)
        ax.set_xlabel("Scattering angle theta_s [deg]")
        if i % 3 == 0:
            ax.set_ylabel("BRDF [1/sr]")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(
            f"Panel {i+1}: secondary = {unit!r}\n"
            f"({spec.label_en})",
            fontsize=10,
        )
        if unit != "theta_s":
            add_secondary_xaxis_mpl(ax, unit, WAVELENGTH_UM)

    # ── Panel 5: reference table (text) ─────────────────────────────────
    ax = axes_flat[5]
    ax.axis("off")
    table_lines = [
        "Direction-cosine family analogy (same info, different units)",
        f"lambda = {WAVELENGTH_UM*1000:.0f} nm, theta_i = {THETA_I_DEG:.0f} deg",
        "",
        "theta_s = 1.0 deg  -> u = 0.0175, k_x = 0.20 rad/um,",
        "                     f = 0.032 1/um, Lambda = 31.5 um",
        "theta_s = 10 deg   -> u = 0.174, k_x = 1.98 rad/um,",
        "                     f = 0.316 1/um, Lambda = 3.17 um",
        "theta_s = 30 deg   -> u = 0.500, k_x = 5.71 rad/um,",
        "                     f = 0.909 1/um, Lambda = 1.10 um",
        "theta_s = 60 deg   -> u = 0.866, k_x = 9.89 rad/um,",
        "                     f = 1.575 1/um, Lambda = 0.64 um",
        "theta_s = 89.9 deg -> u = 1.000, k_x = 11.42 rad/um,",
        "                     f = 1.818 1/um, Lambda = 0.55 um",
        "",
        "Physical meaning:",
        "  u, k_x, f : direction / wave representation (all equiv)",
        "  Lambda    : grating period in surface (AG bump pitch)",
    ]
    y = 0.95
    for line in table_lines:
        ax.text(
            0.05, y, line,
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            verticalalignment="top",
        )
        y -= 0.055

    fig.suptitle(
        f"BSDF 1D profile: same curve with 5 different secondary x-axis units  "
        f"(SphericalArray R=5um P=10um Random, lambda={WAVELENGTH_UM*1000:.0f}nm)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
