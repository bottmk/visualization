"""Multiple theta_i BSDF comparison: absolute vs relative vs corrected coords.

Generates a 3-panel PNG to demonstrate:
  (A) Absolute (u, BSDF): specular peak moves with theta_i
  (B) Relative (u - u_spec, BSDF): peak centered but shape/amplitude differ
  (C) Corrected (u', BSDF * cos(theta_s) / cos^2(theta_i)): profiles mostly overlap
       for smooth surfaces (Rayleigh-Rice regime)

Uses RandomRoughSurface with small Rq (5 nm) = linear regime where corrections
work well.
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

OUT = Path(__file__).parent / "theta_i_relative_comparison.png"

# ── 入力条件 ─────────────────────────────────────────────────────────
WAVELENGTH_UM = 0.55
N1 = 1.0
N2 = 1.5
THETA_I_LIST = [0.0, 20.0, 45.0, 60.0]

# 粗さパラメータ。
# Rq を大きくすると実際の表面散乱が DFT のスペクトル漏れを上回り、
# 理論通り「θ_i 間で (u-u_spec, BSDF) が重なる」挙動が観察できる。
# Rq=5nm だと漏れが優勢で重ならない（FFT 法の数値限界）。
RQ_UM = 0.05      # 50nm（中粗さ：Rayleigh-Rice 境界付近）
LC_UM = 2.0
FRACTAL_DIM = 2.5
GRID_SIZE = 1024  # 大きめのグリッドで漏れ分布を相対的に薄める
PIXEL_UM = 0.25


def extract_phi0_slice(u: np.ndarray, v: np.ndarray, bsdf: np.ndarray):
    """Extract phi=0 (v=0) slice, u >= 0 half, sorted ascending."""
    u_axis = u[:, 0]
    bsdf_slice = bsdf[:, 0]
    half = (u_axis >= 0) & (np.abs(u_axis) <= 1.0)
    u_pos = u_axis[half]
    order = np.argsort(u_pos)
    return u_pos[order], np.maximum(bsdf_slice[half][order], 1e-10)


def main() -> None:
    surface = RandomRoughSurface(
        grid_size=GRID_SIZE, pixel_size_um=PIXEL_UM,
        rq_um=RQ_UM, lc_um=LC_UM, fractal_dim=FRACTAL_DIM, seed=42,
    )
    hm = surface.get_height_map()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(THETA_I_LIST)))

    for ti_deg, color in zip(THETA_I_LIST, colors):
        u_grid, v_grid, bsdf = compute_bsdf_fft(
            hm, wavelength_um=WAVELENGTH_UM,
            theta_i_deg=ti_deg, phi_i_deg=0.0,
            n1=N1, n2=N2, polarization="Unpolarized", is_btdf=False,
        )
        u, y = extract_phi0_slice(u_grid, v_grid, bsdf)
        u_spec = N1 * np.sin(np.deg2rad(ti_deg))
        cos_ti = np.cos(np.deg2rad(ti_deg))
        theta_s = np.rad2deg(np.arcsin(np.clip(u, 0, 1)))
        cos_ts = np.sqrt(np.clip(1 - u**2, 0, 1))

        label = f"theta_i = {ti_deg:.0f} deg"

        # (A) 絶対座標
        axes[0].plot(theta_s, y, color=color, label=label, linewidth=1.5)
        # (B) 相対座標
        u_rel = u - u_spec
        axes[1].plot(u_rel, y, color=color, label=label, linewidth=1.5)
        # (C) 補正後
        y_corrected = y * cos_ts / max(cos_ti ** 2, 1e-6)
        axes[2].plot(u_rel, y_corrected, color=color, label=label, linewidth=1.5)

    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 90)
    axes[0].set_xticks(range(0, 91, 15))
    axes[0].set_xlabel("Scattering angle theta_s [deg]")
    axes[0].set_ylabel("BSDF [1/sr]")
    axes[0].set_title("(A) Absolute coords\npeak shifts with theta_i")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].set_yscale("log")
    axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("u - u_spec  (= sin theta_s - sin theta_i)")
    axes[1].set_ylabel("BSDF [1/sr]")
    axes[1].set_title("(B) Relative coords\npeak at origin; amplitude still differs")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].axvline(0, color="gray", linestyle=":", linewidth=0.8)

    axes[2].set_yscale("log")
    axes[2].set_xlim(-1, 1)
    axes[2].set_xlabel("u - u_spec")
    axes[2].set_ylabel("BSDF * cos(theta_s) / cos^2(theta_i)  [1/sr]")
    axes[2].set_title("(C) Corrected\ncurves mostly overlap for smooth surface")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].axvline(0, color="gray", linestyle=":", linewidth=0.8)

    fig.suptitle(
        f"BSDF vs theta_i  (RandomRough Rq={RQ_UM*1000:.0f}nm, "
        f"lc={LC_UM}um, N={GRID_SIZE}, lambda={WAVELENGTH_UM*1000:.0f}nm, BRDF)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
