"""Parquet スキーマの保存 → 読み込み往復テスト（spec_main.md 9.4 未テスト項目）。

build_dataframe / build_measured_dataframe → save_parquet → load_parquet で
値・dtype・カテゴリ・行順が保持されることを検証する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bsdf_sim.io.parquet_schema import (
    SCHEMA_DTYPES,
    VALID_METHODS,
    VALID_MODES,
    VALID_POLARIZATIONS,
    build_dataframe,
    build_measured_dataframe,
    load_parquet,
    save_parquet,
)


# ── フィクスチャ ──────────────────────────────────────────────────────────────


@pytest.fixture
def sim_df() -> pd.DataFrame:
    """5×5 の UV グリッド（半球内点のみ）から FFT 由来の DataFrame を作る。"""
    u1d = np.linspace(-0.9, 0.9, 5, dtype=np.float32)
    v1d = np.linspace(-0.9, 0.9, 5, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u1d, v1d)
    rng = np.random.default_rng(42)
    bsdf = rng.uniform(0.01, 10.0, size=u_grid.shape).astype(np.float32)

    return build_dataframe(
        u_grid=u_grid,
        v_grid=v_grid,
        bsdf=bsdf,
        method="FFT",
        theta_i_deg=20.0,
        phi_i_deg=0.0,
        wavelength_um=0.525,
        polarization="Unpolarized",
        is_btdf=False,
        log_rmse=0.123,
    )


@pytest.fixture
def measured_df() -> pd.DataFrame:
    """10 点の実測風 1D データから DataFrame を作る。"""
    rng = np.random.default_rng(7)
    theta_s = rng.uniform(0, 80, size=10).astype(np.float32)
    phi_s = rng.uniform(0, 360, size=10).astype(np.float32)
    bsdf_vals = rng.uniform(1e-4, 1.0, size=10).astype(np.float32)

    return build_measured_dataframe(
        theta_s_deg=theta_s,
        phi_s_deg=phi_s,
        bsdf_values=bsdf_vals,
        theta_i_deg=30.0,
        phi_i_deg=0.0,
        wavelength_nm=525.0,
        polarization="S",
        is_btdf=False,
    )


# ── 往復テスト ────────────────────────────────────────────────────────────────


class TestParquetRoundTrip:
    """save_parquet → load_parquet で DataFrame が一致することを検証。"""

    def test_sim_df_roundtrip_equal(self, tmp_path: Path, sim_df: pd.DataFrame) -> None:
        """シミュレーション DataFrame の往復で全行全列一致。"""
        path = tmp_path / "sim.parquet"
        save_parquet(sim_df, path)
        loaded = load_parquet(path)

        pd.testing.assert_frame_equal(sim_df, loaded, check_categorical=True)

    def test_measured_df_roundtrip_equal(self, tmp_path: Path, measured_df: pd.DataFrame) -> None:
        """実測 DataFrame の往復で全行全列一致。"""
        path = tmp_path / "meas.parquet"
        save_parquet(measured_df, path)
        loaded = load_parquet(path)

        pd.testing.assert_frame_equal(measured_df, loaded, check_categorical=True)

    def test_dtypes_preserved(self, tmp_path: Path, sim_df: pd.DataFrame) -> None:
        """SCHEMA_DTYPES で定義された dtype が往復後も一致。"""
        path = tmp_path / "dtype.parquet"
        save_parquet(sim_df, path)
        loaded = load_parquet(path)

        for col, expected_dtype in SCHEMA_DTYPES.items():
            assert col in loaded.columns, f"カラム {col} が欠落"
            actual = str(loaded[col].dtype)
            if expected_dtype == "category":
                assert actual == "category", f"{col}: category 期待 → {actual}"
            else:
                assert actual == expected_dtype, f"{col}: {expected_dtype} 期待 → {actual}"

    def test_categorical_values_preserved(self, tmp_path: Path, sim_df: pd.DataFrame) -> None:
        """カテゴリカル列の categories 定義が完全に保持される。"""
        path = tmp_path / "cat.parquet"
        save_parquet(sim_df, path)
        loaded = load_parquet(path)

        assert list(loaded["polarization"].cat.categories) == VALID_POLARIZATIONS
        assert list(loaded["mode"].cat.categories) == VALID_MODES
        assert list(loaded["method"].cat.categories) == VALID_METHODS

    def test_numeric_values_bitwise_equal(self, tmp_path: Path, sim_df: pd.DataFrame) -> None:
        """float32 数値が往復で bitwise に一致（snappy 圧縮はロスレス）。"""
        path = tmp_path / "numeric.parquet"
        save_parquet(sim_df, path)
        loaded = load_parquet(path)

        for col in ["u", "v", "theta_s_deg", "phi_s_deg", "bsdf", "log_rmse"]:
            original = sim_df[col].to_numpy()
            reloaded = loaded[col].to_numpy()
            # log_rmse 以外は NaN を含まないので exact equality
            np.testing.assert_array_equal(
                original, reloaded, err_msg=f"{col} が bitwise 一致しない"
            )

    def test_row_order_preserved(self, tmp_path: Path, sim_df: pd.DataFrame) -> None:
        """行の並び順が往復で維持される。"""
        path = tmp_path / "order.parquet"
        save_parquet(sim_df, path)
        loaded = load_parquet(path)

        assert len(loaded) == len(sim_df)
        # u, v の組で同一性を確認（同じ値が同じ行位置にあるか）
        np.testing.assert_array_equal(loaded["u"].to_numpy(), sim_df["u"].to_numpy())
        np.testing.assert_array_equal(loaded["v"].to_numpy(), sim_df["v"].to_numpy())

    def test_nan_log_rmse_preserved(self, tmp_path: Path) -> None:
        """log_rmse=None（→ NaN）が往復後も NaN で保たれる。"""
        u_grid, v_grid = np.meshgrid(
            np.linspace(-0.5, 0.5, 3, dtype=np.float32),
            np.linspace(-0.5, 0.5, 3, dtype=np.float32),
        )
        bsdf = np.ones_like(u_grid, dtype=np.float32)
        df = build_dataframe(
            u_grid=u_grid,
            v_grid=v_grid,
            bsdf=bsdf,
            method="PSD",
            theta_i_deg=0.0,
            phi_i_deg=0.0,
            wavelength_um=0.465,
            polarization="S",
            log_rmse=None,
        )
        path = tmp_path / "nan.parquet"
        save_parquet(df, path)
        loaded = load_parquet(path)

        assert loaded["log_rmse"].isna().all()

    def test_btdf_mode_preserved(self, tmp_path: Path) -> None:
        """is_btdf=True で mode='BTDF' が往復で保たれる。"""
        u_grid, v_grid = np.meshgrid(
            np.linspace(-0.5, 0.5, 3, dtype=np.float32),
            np.linspace(-0.5, 0.5, 3, dtype=np.float32),
        )
        df = build_dataframe(
            u_grid=u_grid,
            v_grid=v_grid,
            bsdf=np.ones_like(u_grid, dtype=np.float32),
            method="FFT",
            theta_i_deg=20.0,
            phi_i_deg=0.0,
            wavelength_um=0.525,
            polarization="Unpolarized",
            is_btdf=True,
        )
        path = tmp_path / "btdf.parquet"
        save_parquet(df, path)
        loaded = load_parquet(path)

        assert (loaded["mode"] == "BTDF").all()

    def test_multiple_roundtrips_stable(self, tmp_path: Path, sim_df: pd.DataFrame) -> None:
        """save → load → save → load を繰り返しても DataFrame が変化しない。"""
        path1 = tmp_path / "r1.parquet"
        path2 = tmp_path / "r2.parquet"

        save_parquet(sim_df, path1)
        loaded1 = load_parquet(path1)
        save_parquet(loaded1, path2)
        loaded2 = load_parquet(path2)

        pd.testing.assert_frame_equal(loaded1, loaded2)

    def test_save_creates_parent_dir(self, tmp_path: Path, sim_df: pd.DataFrame) -> None:
        """save_parquet は親ディレクトリを自動作成する。"""
        path = tmp_path / "nested" / "deep" / "out.parquet"
        assert not path.parent.exists()

        save_parquet(sim_df, path)

        assert path.exists()
        loaded = load_parquet(path)
        pd.testing.assert_frame_equal(sim_df, loaded)

    def test_hemisphere_filter_applied(self, tmp_path: Path) -> None:
        """半球外（u² + v² > 1）の点は保存されず、読み込んでも含まれない。"""
        u1d = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
        v1d = np.linspace(-1.5, 1.5, 7, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(u1d, v1d)
        bsdf = np.ones_like(u_grid, dtype=np.float32)

        df = build_dataframe(
            u_grid=u_grid,
            v_grid=v_grid,
            bsdf=bsdf,
            method="FFT",
            theta_i_deg=0.0,
            phi_i_deg=0.0,
            wavelength_um=0.525,
            polarization="Unpolarized",
        )
        path = tmp_path / "hemi.parquet"
        save_parquet(df, path)
        loaded = load_parquet(path)

        uv_r2 = loaded["u"].to_numpy() ** 2 + loaded["v"].to_numpy() ** 2
        assert (uv_r2 <= 1.0 + 1e-6).all(), "半球外の点が含まれている"
