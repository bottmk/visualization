"""ランダム粗面モデル（RandomRoughSurface）。

FFTフィルタ法を用いて、指定した RMS粗さ・相関長・フラクタル次元を持つ
Gaussian 統計のランダム粗面を生成する。
"""

from __future__ import annotations

import numpy as np

from .base import BaseSurfaceModel


class RandomRoughSurface(BaseSurfaceModel):
    """ランダム粗面モデル。

    FFTフィルタ法（スペクトル法）により、以下の統計パラメータを持つ
    ガウス統計ランダム粗面を生成する。

    Args:
        rq_um: RMS粗さ [μm]（例: 5nm = 0.005μm）
        lc_um: 相関長 [μm]
        fractal_dim: フラクタル次元（2.0〜3.0）。2.0 は白色スペクトル、3.0 は最も滑らか。
        grid_size: 本計算用グリッドサイズ（デフォルト: 4096）
        pixel_size_um: ピクセルサイズ [μm]（デフォルト: 0.25μm）
        seed: 乱数シード（None の場合はランダム）
    """

    def __init__(
        self,
        rq_um: float,
        lc_um: float,
        fractal_dim: float = 2.5,
        grid_size: int = 4096,
        pixel_size_um: float = 0.25,
        seed: int | None = None,
    ) -> None:
        super().__init__(grid_size=grid_size, pixel_size_um=pixel_size_um)
        if rq_um <= 0:
            raise ValueError(f"rq_um は正の値でなければならない。値={rq_um}")
        if lc_um <= 0:
            raise ValueError(f"lc_um は正の値でなければならない。値={lc_um}")
        if not (2.0 <= fractal_dim <= 3.0):
            raise ValueError(f"fractal_dim は 2.0〜3.0 の範囲でなければならない。値={fractal_dim}")

        self.rq_um = rq_um
        self.lc_um = lc_um
        self.fractal_dim = fractal_dim
        self.seed = seed

    def _generate(self, grid_size: int, pixel_size_um: float) -> np.ndarray:
        """FFTフィルタ法によるランダム粗面生成。

        アルゴリズム:
        1. ガウス乱数配列を生成
        2. FFT 後、パワースペクトルフィルタを乗算
        3. IFFT で空間域に戻し、RMS粗さを正規化

        フィルタ形状:
        - ガウス型自己相関: PSD ∝ exp(-(f·π·Lc)²) / (空間周波数ゼロ成分を基準)
        - フラクタル成分: 高周波側の傾き ∝ f^(-(2H+1))（H = fractal_dim - 2）

        Args:
            grid_size: 生成するグリッドサイズ
            pixel_size_um: ピクセルサイズ [μm]

        Returns:
            shape (grid_size, grid_size) の高さ配列 [μm]
        """
        rng = np.random.default_rng(self.seed)

        # 空間周波数グリッド [μm⁻¹]
        freq = np.fft.fftfreq(grid_size, d=pixel_size_um)
        fx, fy = np.meshgrid(freq, freq, indexing="ij")
        f_r = np.sqrt(fx**2 + fy**2)

        # PSD フィルタ: ガウス自己相関 × フラクタルロールオフ
        # ガウス相関長成分: exp(-(π·f·Lc)²)
        gaussian_filter = np.exp(-((np.pi * f_r * self.lc_um) ** 2))

        # フラクタル成分: f^(-H) ロールオフ（直流成分を除く）
        # Hurst 指数 H = fractal_dim - 2（fractal_dim=2→H=0、fractal_dim=3→H=1）
        H = self.fractal_dim - 2.0
        # f_r=0 での 0**(-H) による RuntimeWarning を防ぐため安全な値で置換
        f_r_safe = np.where(f_r > 0, f_r, 1.0)
        fractal_filter = np.where(f_r > 0, f_r_safe ** (-H), 1.0)

        # 合成フィルタ（DC成分は1に正規化）
        combined_filter = gaussian_filter * fractal_filter
        dc_value = combined_filter[0, 0]
        if dc_value > 0:
            combined_filter /= dc_value

        # ホワイトノイズを生成してフィルタリング
        noise = rng.standard_normal((grid_size, grid_size))
        noise_fft = np.fft.fft2(noise)
        filtered_fft = noise_fft * np.sqrt(combined_filter)
        surface = np.real(np.fft.ifft2(filtered_fft))

        # RMS粗さを目標値に正規化
        current_rq = np.sqrt(np.mean(surface**2))
        if current_rq > 0:
            surface *= self.rq_um / current_rq

        return surface.astype(np.float32)

    @classmethod
    def from_config(cls, config: dict) -> "RandomRoughSurface":
        """設定辞書からインスタンスを生成する。"""
        surface_cfg = config.get("surface", {})
        rr_cfg = surface_cfg.get("random_rough", {})
        return cls(
            rq_um=rr_cfg["rq_um"],
            lc_um=rr_cfg["lc_um"],
            fractal_dim=rr_cfg.get("fractal_dim", 2.5),
            grid_size=surface_cfg.get("grid_size", 4096),
            pixel_size_um=surface_cfg.get("pixel_size_um", 0.25),
        )
