"""Compare FFT (no Fresnel) / FFT (+Fresnel post-multiply) / PSD (Rayleigh-Rice).

Uses SphericalArraySurface (R=5um, P=10um, Random placement). For each theta_i,
plot BRDF vs scattering angle at phi=0 plane (phi band-averaged) on semilog-y.

Random placement washes out the strong diffraction lattice peaks that a
Hexagonal/Grid placement produces, so the overall envelope of the 3 methods
is visible.

The Fresnel post-multiplication applies R(theta_i) = (|r_s|^2 + |r_p|^2)/2
uniformly to all scattering directions (theta_s-independent heuristic).
PSD implements Elson-Bennett complete Q(theta_i, theta_s) so its angular
shape is more accurate at wide theta_i.

Saves a 1x4 panel PNG (one axis per theta_i).
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
from bsdf_sim.optics.fresnel import fresnel_rs, fresnel_rp
from bsdf_sim.optics.psd_bsdf import compute_bsdf_psd
from bsdf_sim.visualization.profile_extract import slice_phi0

OUT = Path(__file__).parent / "fft_fresnel_vs_psd.png"

WAVELENGTH_UM = 0.55
GRID_SIZE = 1024
PIXEL_UM = 0.15
N1 = 1.0
N2 = 1.5

THETA_I_LIST = [0.0, 20.0, 45.0, 60.0]
FFT_MODE = "tilt"
PLACEMENT = "Random"
V_BAND_BINS = 12        # wider phi band averaging to suppress speckle


def extract_phi0_slice(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bsdf: np.ndarray,
    v_band_bins: int = 5,
):
    """Return (u_axis, bsdf_mean) averaged over |v| within v_band_bins bins.

    Averaging a narrow phi band reduces speckle from the single-realization
    surface so the overall envelope is visible.
    """
    return slice_phi0(
        u_grid, v_grid, bsdf,
        mode="signed", v_band_bins=v_band_bins, floor=1e-20,
    )


def u_to_theta_s_signed(u: np.ndarray) -> np.ndarray:
    """Signed scattering angle [deg] from direction cosine u (phi=0 plane)."""
    u_clipped = np.clip(u, -1.0, 1.0)
    return np.rad2deg(np.arcsin(u_clipped))


def main() -> None:
    print(f"building SphericalArraySurface (R=5um, P=10um, {PLACEMENT}) ...")
    model = SphericalArraySurface(
        radius_um=5.0,
        pitch_um=10.0,
        base_height_um=0.0,
        placement=PLACEMENT,
        overlap_mode="Maximum",
        grid_size=GRID_SIZE,
        pixel_size_um=PIXEL_UM,
        seed=42,
    )
    hm = model.get_height_map()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), sharey=True)

    for ax, ti_deg in zip(axes, THETA_I_LIST):
        print(f"  computing theta_i = {ti_deg:.0f} deg ...")

        u_a, v_a, bsdf_a = compute_bsdf_fft(
            hm,
            wavelength_um=WAVELENGTH_UM,
            theta_i_deg=ti_deg,
            phi_i_deg=0.0,
            n1=N1,
            n2=N2,
            is_btdf=False,
            fft_mode=FFT_MODE,
            apply_fresnel=False,
        )
        u_axis_a, bsdf_a_row = extract_phi0_slice(u_a, v_a, bsdf_a, V_BAND_BINS)

        u_b, v_b, bsdf_b = compute_bsdf_fft(
            hm,
            wavelength_um=WAVELENGTH_UM,
            theta_i_deg=ti_deg,
            phi_i_deg=0.0,
            n1=N1,
            n2=N2,
            is_btdf=False,
            fft_mode=FFT_MODE,
            apply_fresnel=True,
        )
        u_axis_b, bsdf_b_row = extract_phi0_slice(u_b, v_b, bsdf_b, V_BAND_BINS)

        u_c, v_c, bsdf_c = compute_bsdf_psd(
            hm,
            wavelength_um=WAVELENGTH_UM,
            theta_i_deg=ti_deg,
            phi_i_deg=0.0,
            n1=N1,
            n2=N2,
            polarization="Unpolarized",
            is_btdf=False,
        )
        u_axis_c, bsdf_c_row = extract_phi0_slice(u_c, v_c, bsdf_c, V_BAND_BINS)

        rs = abs(complex(fresnel_rs(ti_deg, N1, N2))) ** 2
        rp = abs(complex(fresnel_rp(ti_deg, N1, N2))) ** 2
        fresnel_R = 0.5 * (rs + rp)

        ax.plot(
            u_to_theta_s_signed(u_axis_a), bsdf_a_row,
            linewidth=1.6, color="#1f77b4", label="FFT (no Fresnel)",
        )
        ax.plot(
            u_to_theta_s_signed(u_axis_b), bsdf_b_row,
            linewidth=1.6, color="#d62728", label="FFT (+Fresnel)",
        )
        ax.plot(
            u_to_theta_s_signed(u_axis_c), bsdf_c_row,
            linewidth=1.6, color="#2ca02c", linestyle="--",
            label="PSD (Rayleigh-Rice)",
        )

        ax.axvline(ti_deg, color="gray", linestyle=":", linewidth=0.8,
                   label="specular (theta_s = theta_i)")
        ax.set_yscale("log")
        ax.set_xlim(-90, 90)
        ax.set_ylim(1e-4, 1e5)
        ax.set_xlabel("theta_s [deg]  (phi=0 plane)")
        ax.set_title(
            f"theta_i = {ti_deg:.0f} deg\n"
            f"R(theta_i) = {fresnel_R:.4f}"
        )
        ax.grid(True, which="both", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("BRDF [sr^-1]  (phi band-averaged)")

    fig.suptitle(
        f"SphericalArray (R=5um, P=10um, {PLACEMENT}) BRDF  "
        f"lambda={WAVELENGTH_UM*1000:.0f}nm, n1={N1}, n2={N2}, "
        f"FFT mode={FFT_MODE}",
        fontsize=12,
    )

    # Shared legend below all axes.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=10,
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
