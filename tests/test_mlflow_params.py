"""MLflow 用 params 構築ヘルパ `build_run_params` のテスト。

spec_main.md Section 6.2 の params 仕様:
  - surface_design / surface_measured / bsdf_measured（短縮名、排他的）
  - shape_data_path / bsdf_data_path（該当時のみ）
  - 形状パラメータ（モデル別）
  - sim 条件（wavelength_um / theta_i_deg / ... 多条件は JSON list）
  - sim 専用条件（fft_mode / apply_fresnel）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bsdf_sim.io.config_loader import BSDFConfig
from bsdf_sim.optimization.mlflow_logger import (
    _short_name,
    _stringify,
    build_run_params,
)


# ── 共通ユーティリティのテスト ──────────────────────────────────────────────


class TestShortName:
    """クラス名 → 短縮名変換ルール（Surface/BsdfReader 末尾 + Device 先頭除去）。"""

    @pytest.mark.parametrize("full,short", [
        ("RandomRoughSurface", "RandomRough"),
        ("SphericalArraySurface", "SphericalArray"),
        ("MeasuredSurface", "Measured"),
        ("DeviceVk6Surface", "Vk6"),
        ("LightToolsBsdfReader", "LightTools"),
    ])
    def test_known_names(self, full, short):
        assert _short_name(full) == short

    def test_plain_name_untouched(self):
        """末尾 suffix も Device プレフィックスも無ければ変換されない。"""
        assert _short_name("CustomModel") == "CustomModel"


class TestStringify:
    """MLflow params 値の str 化（単一要素はスカラ、複数要素は JSON list）。"""

    def test_scalar_passes_through(self):
        assert _stringify(0.525) == "0.525"
        assert _stringify("BRDF") == "BRDF"
        assert _stringify(True) == "True"

    def test_singleton_list_is_scalar(self):
        """要素が 1 個のリストはスカラ文字列化（JSON 化しない）。"""
        assert _stringify([0.525]) == "0.525"

    def test_multi_element_list_is_json(self):
        """要素が 2 個以上のリストは JSON 文字列化。"""
        result = _stringify([0.465, 0.525, 0.630])
        assert json.loads(result) == [0.465, 0.525, 0.630]

    def test_tuple_treated_as_list(self):
        result = _stringify((1, 2, 3))
        assert json.loads(result) == [1, 2, 3]


# ── build_run_params の統合テスト ──────────────────────────────────────────


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def base_sim():
    return {
        "simulation": {
            "wavelength_um": 0.525,
            "theta_i_deg": 0.0,
            "phi_i_deg": 0.0,
            "n1": 1.0,
            "n2": 1.5,
            "polarization": "Unpolarized",
        },
    }


class TestBuildRunParamsRandomRough:
    """RandomRoughSurface の params 登録内容。"""

    def test_surface_design_short_name(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
            "grid_size": 512,
            "pixel_size_um": 0.25,
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert params["surface_design"] == "RandomRough"
        assert "surface_measured" not in params

    def test_shape_params_registered(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert params["rq_um"] == "0.005"
        assert params["lc_um"] == "2.0"
        assert params["fractal_dim"] == "2.5"

    def test_no_shape_data_path(self, tmp_path, base_sim):
        """RandomRough は形状データファイルを持たない → shape_data_path 無し。"""
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert "shape_data_path" not in params


class TestBuildRunParamsSphericalArray:
    """SphericalArraySurface の params 登録内容。"""

    def test_surface_design_and_shape_params(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "SphericalArraySurface",
            "spherical_array": {
                "radius_um": 50.0,
                "pitch_um": 100.0,
                "base_height_um": 0.5,
                "placement": "Hexagonal",
                "overlap_mode": "Maximum",
            },
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert params["surface_design"] == "SphericalArray"
        assert params["radius_um"] == "50.0"
        assert params["pitch_um"] == "100.0"
        assert params["base_height_um"] == "0.5"
        assert params["placement"] == "Hexagonal"
        assert params["overlap_mode"] == "Maximum"
        # RandomRough 固有キーは入らない
        assert "rq_um" not in params


class TestBuildRunParamsMeasuredSurface:
    """MeasuredSurface（形状測定系）の params 登録内容。"""

    def test_surface_measured_and_shape_data_path(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "MeasuredSurface",
            "measured": {
                "path": "sample.csv",
                "padding": "smooth_tile",
                "source_pixel_size_um": 0.5,
                "height_unit": "nm",
                "leveling": True,
            },
            "grid_size": 4096,
            "pixel_size_um": 0.25,
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert params["surface_measured"] == "Measured"
        assert "surface_design" not in params
        assert params["shape_data_path"] == "sample.csv"
        assert params["padding"] == "smooth_tile"
        assert params["source_pixel_size_um"] == "0.5"
        assert params["height_unit"] == "nm"
        assert params["leveling"] == "True"
        assert params["grid_size"] == "4096"
        assert params["pixel_size_um"] == "0.25"


class TestBuildRunParamsSimConditions:
    """sim 条件（実測 BSDF 条件と共通語彙、多条件は JSON list）。"""

    def test_single_condition_scalar(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        # 単一条件はスカラ表示
        assert params["wavelength_um"] == "0.525"
        assert params["theta_i_deg"] == "0.0"
        assert params["phi_i_deg"] == "0.0"
        assert params["polarization"] == "Unpolarized"
        assert params["n1"] == "1.0"
        assert params["n2"] == "1.5"

    def test_multi_condition_json_list(self, tmp_path):
        cfg_dict = {
            "simulation": {
                "wavelength_um": [0.465, 0.525, 0.630],
                "theta_i_deg": [0.0, 30.0, 60.0],
                "mode": ["BRDF", "BTDF"],
                "phi_i_deg": 0.0,
                "n1": 1.0,
                "n2": 1.5,
                "polarization": "Unpolarized",
            },
            "surface": {
                "model": "RandomRoughSurface",
                "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
            },
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, cfg_dict))
        params = build_run_params(cfg)
        # JSON 文字列として保存される
        assert json.loads(params["wavelength_um"]) == [0.465, 0.525, 0.630]
        assert json.loads(params["theta_i_deg"]) == [0.0, 30.0, 60.0]
        assert json.loads(params["mode"]) == ["BRDF", "BTDF"]


class TestBuildRunParamsFftSpecific:
    """sim 専用条件（fft_mode / apply_fresnel）。"""

    def test_default_fft_params(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert params["fft_mode"] == "tilt"  # 既定値
        assert params["apply_fresnel"] == "False"

    def test_custom_fft_params(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        base_sim["fft"] = {"mode": "output_shift", "apply_fresnel": True}
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert params["fft_mode"] == "output_shift"
        assert params["apply_fresnel"] == "True"


class TestBuildRunParamsMeasuredBsdf:
    """measured_bsdf.path 指定時の bsdf_measured / bsdf_data_path 登録。"""

    def test_no_measured_bsdf(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert "bsdf_measured" not in params
        assert "bsdf_data_path" not in params

    def test_measured_bsdf_path_registered(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        sample_bsdf = Path(__file__).parent.parent / "sample_inputs" / "BRDF_BTDF_LightTools.bsdf"
        if not sample_bsdf.exists():
            pytest.skip("sample_inputs/BRDF_BTDF_LightTools.bsdf が存在しない")
        base_sim["measured_bsdf"] = {"path": str(sample_bsdf)}
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert params["bsdf_data_path"] == str(sample_bsdf)
        # LightTools リーダーが auto-detect される
        assert params["bsdf_measured"] == "LightTools"

    def test_measured_bsdf_coexists_with_surface_design(self, tmp_path, base_sim):
        """surface_design + bsdf_measured の同時登録（比較 run の典型）。"""
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        sample_bsdf = Path(__file__).parent.parent / "sample_inputs" / "BRDF_BTDF_LightTools.bsdf"
        if not sample_bsdf.exists():
            pytest.skip("sample_inputs/BRDF_BTDF_LightTools.bsdf が存在しない")
        base_sim["measured_bsdf"] = {"path": str(sample_bsdf)}
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg)
        assert params["surface_design"] == "RandomRough"
        assert params["bsdf_measured"] == "LightTools"
        # surface_measured は入らない（RandomRough は MeasuredSurface ではないので）
        assert "surface_measured" not in params


class TestBuildRunParamsExtra:
    """`extra` 引数による追加 params（optimize で trial 値を注入する用途）。"""

    def test_extra_params_are_merged(self, tmp_path, base_sim):
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg, extra={"trial_number": 42})
        assert params["trial_number"] == "42"
        # 既存 key は保持
        assert params["surface_design"] == "RandomRough"

    def test_extra_overrides_existing_key(self, tmp_path, base_sim):
        """extra で既存 key を上書きできる（optimize で trial 値が config を上書きする用途）。"""
        base_sim["surface"] = {
            "model": "RandomRoughSurface",
            "random_rough": {"rq_um": 0.005, "lc_um": 2.0, "fractal_dim": 2.5},
        }
        cfg = BSDFConfig.from_file(_write_cfg(tmp_path, base_sim))
        params = build_run_params(cfg, extra={"rq_um": 0.01})
        assert params["rq_um"] == "0.01"
