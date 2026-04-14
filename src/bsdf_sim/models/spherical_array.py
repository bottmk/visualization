"""球面アレイモデル（SphericalArraySurface）。

レンズ配置アルゴリズム（Grid / Hexagonal / Random / PoissonDisk）と
重なり処理（Maximum / Additive）を組み合わせた球面レンズアレイ表面を生成する。
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .base import BaseSurfaceModel


# ── レンズ配置アルゴリズム ────────────────────────────────────────────────────

def _place_grid(physical_size_um: float, pitch_um: float) -> np.ndarray:
    """正方格子配置。

    Returns:
        shape (N, 2) の中心座標配列 [μm]
    """
    coords = np.arange(0, physical_size_um, pitch_um) + pitch_um / 2
    xx, yy = np.meshgrid(coords, coords)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _place_hexagonal(physical_size_um: float, pitch_um: float) -> np.ndarray:
    """六方格子配置。

    Returns:
        shape (N, 2) の中心座標配列 [μm]
    """
    centers = []
    row_pitch = pitch_um * np.sqrt(3) / 2
    row = 0
    y = pitch_um / 2
    while y < physical_size_um:
        x_offset = (pitch_um / 2) if (row % 2 == 1) else 0.0
        x = x_offset + pitch_um / 2
        while x < physical_size_um:
            centers.append([x, y])
            x += pitch_um
        y += row_pitch
        row += 1
    return np.array(centers, dtype=np.float64)


def _place_random(
    physical_size_um: float,
    pitch_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """ランダム配置（密度は六方格子と同等）。

    Returns:
        shape (N, 2) の中心座標配列 [μm]
    """
    area = physical_size_um**2
    unit_area = (pitch_um**2) * np.sqrt(3) / 2
    n_lenses = max(1, int(area / unit_area))
    xy = rng.uniform(0, physical_size_um, size=(n_lenses, 2))
    return xy


def _place_poisson_disk(
    physical_size_um: float,
    pitch_um: float,
    rng: np.random.Generator,
    max_attempts: int = 30,
) -> np.ndarray:
    """ポアソンディスクサンプリング配置（最小間隔 pitch_um を保持）。

    Bridson アルゴリズムの簡易実装。

    Returns:
        shape (N, 2) の中心座標配列 [μm]
    """
    min_dist = pitch_um
    cell_size = min_dist / np.sqrt(2)
    grid_n = int(np.ceil(physical_size_um / cell_size))
    grid: dict[tuple[int, int], np.ndarray] = {}

    def grid_key(pt: np.ndarray) -> tuple[int, int]:
        return (int(pt[0] / cell_size), int(pt[1] / cell_size))

    def is_valid(pt: np.ndarray) -> bool:
        if not (0 <= pt[0] < physical_size_um and 0 <= pt[1] < physical_size_um):
            return False
        gx, gy = grid_key(pt)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                neighbor_key = (gx + dx, gy + dy)
                if neighbor_key in grid:
                    if np.linalg.norm(pt - grid[neighbor_key]) < min_dist:
                        return False
        return True

    first = rng.uniform(0, physical_size_um, size=2)
    grid[grid_key(first)] = first
    active = [first]
    samples = [first]

    while active:
        idx = rng.integers(len(active))
        base = active[idx]
        found = False
        for _ in range(max_attempts):
            r = rng.uniform(min_dist, 2 * min_dist)
            angle = rng.uniform(0, 2 * np.pi)
            candidate = base + np.array([r * np.cos(angle), r * np.sin(angle)])
            if is_valid(candidate):
                grid[grid_key(candidate)] = candidate
                active.append(candidate)
                samples.append(candidate)
                found = True
                break
        if not found:
            active.pop(idx)

    return np.array(samples)


# ── 単一レンズの高さプロファイル ──────────────────────────────────────────────

def _spherical_lens_height(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    radius_um: float,
    base_height_um: float,
) -> np.ndarray:
    """球面キャップの高さプロファイルを計算する。

    Args:
        x, y: グリッド座標 [μm]
        cx, cy: レンズ中心座標 [μm]
        radius_um: 曲率半径 [μm]
        base_height_um: ベース高さ（底面オフセット）[μm]

    Returns:
        高さ配列（レンズ外は 0）[μm]
    """
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    inside = r2 <= radius_um**2
    height = np.zeros_like(x, dtype=np.float32)
    r2_safe = np.where(inside, r2, 0.0)
    height[inside] = (radius_um - np.sqrt(radius_um**2 - r2_safe[inside])) + base_height_um
    return height


# ── メインクラス ──────────────────────────────────────────────────────────────

PlacementAlgorithm = Literal["Grid", "Hexagonal", "Random", "PoissonDisk"]
OverlapMode = Literal["Maximum", "Additive"]


class SphericalArraySurface(BaseSurfaceModel):
    """球面レンズアレイ表面モデル。

    Args:
        radius_um: レンズ曲率半径 [μm]
        pitch_um: 配置ピッチ [μm]（Grid/Hexagonal 用）
        base_height_um: ベース高さ [μm]
        placement: 配置アルゴリズム（'Grid' / 'Hexagonal' / 'Random' / 'PoissonDisk'）
        overlap_mode: 重なり処理（'Maximum'=Max値採用 / 'Additive'=加算）
        grid_size: 本計算用グリッドサイズ（デフォルト: 4096）
        pixel_size_um: ピクセルサイズ [μm]（デフォルト: 0.25μm）
        seed: 乱数シード（Random/PoissonDisk 用）
    """

    def __init__(
        self,
        radius_um: float,
        pitch_um: float,
        base_height_um: float = 0.0,
        placement: PlacementAlgorithm = "Hexagonal",
        overlap_mode: OverlapMode = "Maximum",
        grid_size: int = 4096,
        pixel_size_um: float = 0.25,
        seed: int | None = None,
    ) -> None:
        super().__init__(grid_size=grid_size, pixel_size_um=pixel_size_um)
        if radius_um <= 0:
            raise ValueError(f"radius_um は正の値でなければならない。値={radius_um}")
        if pitch_um <= 0:
            raise ValueError(f"pitch_um は正の値でなければならない。値={pitch_um}")

        self.radius_um = radius_um
        self.pitch_um = pitch_um
        self.base_height_um = base_height_um
        self.placement = placement
        self.overlap_mode = overlap_mode
        self.seed = seed

    def _generate(self, grid_size: int, pixel_size_um: float) -> np.ndarray:
        """球面レンズアレイの高さ配列を生成する。"""
        physical_size_um = grid_size * pixel_size_um
        rng = np.random.default_rng(self.seed)

        # 配置アルゴリズムで中心座標を取得
        if self.placement == "Grid":
            centers = _place_grid(physical_size_um, self.pitch_um)
        elif self.placement == "Hexagonal":
            centers = _place_hexagonal(physical_size_um, self.pitch_um)
        elif self.placement == "Random":
            centers = _place_random(physical_size_um, self.pitch_um, rng)
        elif self.placement == "PoissonDisk":
            centers = _place_poisson_disk(physical_size_um, self.pitch_um, rng)
        else:
            raise ValueError(f"未知の配置アルゴリズム: {self.placement}")

        # グリッド座標
        coords = (np.arange(grid_size) + 0.5) * pixel_size_um
        x, y = np.meshgrid(coords, coords, indexing="ij")

        # 各レンズの高さを合成
        if self.overlap_mode == "Maximum":
            surface = np.zeros((grid_size, grid_size), dtype=np.float32)
            for cx, cy in centers:
                lens_h = _spherical_lens_height(x, y, cx, cy, self.radius_um, self.base_height_um)
                surface = np.maximum(surface, lens_h)
        elif self.overlap_mode == "Additive":
            surface = np.zeros((grid_size, grid_size), dtype=np.float32)
            for cx, cy in centers:
                lens_h = _spherical_lens_height(x, y, cx, cy, self.radius_um, self.base_height_um)
                surface += lens_h
        else:
            raise ValueError(f"未知の重なり処理: {self.overlap_mode}")

        return surface

    @classmethod
    def from_config(cls, config: dict) -> "SphericalArraySurface":
        """設定辞書からインスタンスを生成する。"""
        surface_cfg = config.get("surface", {})
        sa_cfg = surface_cfg.get("spherical_array", {})
        return cls(
            radius_um=sa_cfg["radius_um"],
            pitch_um=sa_cfg.get("pitch_um", sa_cfg.get("radius_um", 50.0)),
            base_height_um=sa_cfg.get("base_height_um", 0.0),
            placement=sa_cfg.get("placement", "Hexagonal"),
            overlap_mode=sa_cfg.get("overlap_mode", "Maximum"),
            grid_size=surface_cfg.get("grid_size", 4096),
            pixel_size_um=surface_cfg.get("pixel_size_um", 0.25),
        )
