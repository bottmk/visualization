"""dashboard の 1D 重ね描きを matplotlib で再現して PNG として保存するデモ。

sample_inputs/config_comp_meas_bsdf.yaml の SphericalArraySurface を使用し、
LightTools 実測 BSDF（465nm, θ_i=0°, BRDF）と重ねた図を書き出す。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from bsdf_sim.visualization.constants import BSDF_LOG_FLOOR_DEFAULT
from bsdf_sim.visualization.dynamicmap import create_dashboard_from_config
from bsdf_sim.visualization.profile_extract import slice_phi0

CFG = Path(__file__).parent.parent / "sample_inputs" / "config_comp_meas_bsdf.yaml"
OUT = Path(__file__).parent / "dashboard_demo_overlay.png"


def main() -> None:
    dash = create_dashboard_from_config(str(CFG), preview_grid_size_idle=256)

    # SphericalArray の BSDF を計算（dashboard の _compute_preview と同じパス）
    u, v, bsdf = dash._compute_preview(
        radius_um=5.0, pitch_um=10.0, base_height_um=0.0,
        placement="Hexagonal", overlap_mode="Maximum", grid_size=256,
    )

    u_pos, y_sim = slice_phi0(
        u, v, bsdf, mode="positive", floor=BSDF_LOG_FLOOR_DEFAULT,
    )
    x_sim = np.rad2deg(np.arcsin(np.clip(u_pos, 0, 1)))

    prof = dash._measured_profile()
    assert prof is not None, "実測プロファイルが取れていない"
    x_meas, y_meas = prof

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_sim, y_sim, color="blue", linewidth=2, label="FFT (sim)")
    ax.scatter(
        x_meas, y_meas, s=40, facecolors="black", edgecolors="white",
        linewidths=0.8, label="Measured", zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 90)
    ax.set_xticks(range(0, 91, 10))
    ax.set_xticks(range(0, 91, 5), minor=True)
    ax.set_xlabel("Scattering angle theta_s [deg]")
    ax.set_ylabel("BSDF [1/sr]")
    ax.set_title(
        "BSDF dashboard demo (N=256)\n"
        "SphericalArraySurface (R=5um, P=10um, Hexagonal) vs LightTools 465nm BRDF"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"saved: {OUT}")
    print(f"sim points: {len(x_sim)}  measured points: {len(x_meas)}")


if __name__ == "__main__":
    main()
