"""BSDFConfig の設定読み込み・プリセット解決・バリデーションのテスト。"""

import pytest

from bsdf_sim.io.config_loader import BSDFConfig


# ── 最小有効設定 ──────────────────────────────────────────────────────────────

_MINIMAL_CFG = {
    "simulation": {
        "wavelength_um": 0.55,
        "theta_i_deg": 0.0,
        "phi_i_deg": 0.0,
        "n1": 1.0,
        "n2": 1.5,
        "polarization": "Unpolarized",
    },
    "surface": {"model": "RandomRoughSurface"},
    "error_metrics": {"bsdf_floor": 1e-6},
}


class TestBSDFConfigProperties:
    def test_wavelength(self):
        cfg = BSDFConfig(_MINIMAL_CFG)
        assert cfg.wavelength_um == pytest.approx(0.55)

    def test_theta_i(self):
        cfg = BSDFConfig(_MINIMAL_CFG)
        assert cfg.theta_i_deg == pytest.approx(0.0)

    def test_n1_n2(self):
        cfg = BSDFConfig(_MINIMAL_CFG)
        assert cfg.n1 == pytest.approx(1.0)
        assert cfg.n2 == pytest.approx(1.5)

    def test_polarization(self):
        cfg = BSDFConfig(_MINIMAL_CFG)
        assert cfg.polarization == "Unpolarized"

    def test_bsdf_floor(self):
        cfg = BSDFConfig(_MINIMAL_CFG)
        assert cfg.bsdf_floor == pytest.approx(1e-6)

    def test_is_brdf(self):
        cfg = BSDFConfig(_MINIMAL_CFG)
        assert cfg.is_btdf is False

    def test_is_btdf(self):
        raw = {**_MINIMAL_CFG, "simulation": {**_MINIMAL_CFG["simulation"], "theta_i_deg": 150.0}}
        cfg = BSDFConfig(raw)
        assert cfg.is_btdf is True

    def test_theta_i_effective_btdf(self):
        """BTDF モード（theta_i=150°）では有効入射角 = 30°。"""
        raw = {**_MINIMAL_CFG, "simulation": {**_MINIMAL_CFG["simulation"], "theta_i_deg": 150.0}}
        cfg = BSDFConfig(raw)
        assert cfg.theta_i_effective_deg == pytest.approx(30.0)


class TestBSDFConfigValidation:
    def test_theta_90_raises(self):
        raw = {**_MINIMAL_CFG, "simulation": {**_MINIMAL_CFG["simulation"], "theta_i_deg": 90.0}}
        with pytest.raises(ValueError, match="90°"):
            BSDFConfig(raw)

    def test_invalid_polarization_raises(self):
        raw = {**_MINIMAL_CFG, "simulation": {**_MINIMAL_CFG["simulation"], "polarization": "X"}}
        with pytest.raises(ValueError, match="polarization"):
            BSDFConfig(raw)

    def test_zero_bsdf_floor_raises(self):
        raw = {**_MINIMAL_CFG, "error_metrics": {"bsdf_floor": 0.0}}
        with pytest.raises(ValueError, match="bsdf_floor"):
            BSDFConfig(raw)

    def test_negative_bsdf_floor_raises(self):
        raw = {**_MINIMAL_CFG, "error_metrics": {"bsdf_floor": -1e-6}}
        with pytest.raises(ValueError, match="bsdf_floor"):
            BSDFConfig(raw)


class TestPresetResolution:
    """Sparkle プリセット解決のテスト。"""

    def _make_cfg_with_sparkle(self, preset_name: str) -> BSDFConfig:
        raw = {
            **_MINIMAL_CFG,
            "metrics": {
                "sparkle": {
                    "enabled": True,
                    "viewing": {"preset": preset_name},
                    "display": {"preset": "fhd_smartphone"},
                }
            },
        }
        return BSDFConfig(raw)

    def test_smartphone_preset_distance(self):
        cfg = self._make_cfg_with_sparkle("smartphone")
        sparkle = cfg.metrics["sparkle"]
        assert sparkle["viewing"]["distance_mm"] == pytest.approx(300.0)

    def test_tablet_preset_distance(self):
        cfg = self._make_cfg_with_sparkle("tablet")
        sparkle = cfg.metrics["sparkle"]
        assert sparkle["viewing"]["distance_mm"] == pytest.approx(350.0)

    def test_illumination_preset_is_ignored(self):
        """illumination は simulate ループで使われないため resolved から削除される。"""
        raw = {
            **_MINIMAL_CFG,
            "metrics": {
                "sparkle": {
                    "enabled": True,
                    "viewing": {"preset": "smartphone"},
                    "display": {"preset": "fhd_smartphone"},
                    "illumination": {"preset": "green"},  # 指定しても無視される
                }
            },
        }
        cfg = BSDFConfig(raw)
        sparkle = cfg.metrics["sparkle"]
        assert "illumination" not in sparkle

    def test_unknown_preset_raises(self):
        raw = {
            **_MINIMAL_CFG,
            "metrics": {
                "sparkle": {
                    "viewing": {"preset": "unknown_device"},
                    "display": {"preset": "fhd_smartphone"},
                }
            },
        }
        with pytest.raises(ValueError, match="未知のプリセット"):
            BSDFConfig(raw)

    def test_manual_override_takes_priority(self):
        """個別数値指定はプリセットより優先される。"""
        raw = {
            **_MINIMAL_CFG,
            "metrics": {
                "sparkle": {
                    "viewing": {"preset": "smartphone", "distance_mm": 500.0},
                    "display": {"preset": "fhd_smartphone"},
                }
            },
        }
        cfg = BSDFConfig(raw)
        assert cfg.metrics["sparkle"]["viewing"]["distance_mm"] == pytest.approx(500.0)


class TestFromFile:
    def test_from_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BSDFConfig.from_file(tmp_path / "nonexistent.yaml")

    def test_from_file_loads_yaml(self, tmp_path):
        import yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(_MINIMAL_CFG), encoding="utf-8")
        cfg = BSDFConfig.from_file(config_path)
        assert cfg.wavelength_um == pytest.approx(0.55)


# ── 多条件サポート（案 2-B: スカラ/list 両対応）───────────────────────────────


class TestMultiConditionSupport:
    """wavelength_um / theta_i_deg / mode のスカラ・list 両対応。"""

    def _base(self, **overrides):
        cfg = {
            "simulation": {
                "phi_i_deg": 0.0,
                "n1": 1.0,
                "n2": 1.5,
                "polarization": "Unpolarized",
                **overrides,
            },
            "surface": {"model": "RandomRoughSurface"},
            "error_metrics": {"bsdf_floor": 1e-6},
        }
        return BSDFConfig(cfg)

    # ── wavelengths_um ──────────────────────────────────────────────────

    def test_wavelength_scalar_normalized_to_list(self):
        cfg = self._base(wavelength_um=0.55, theta_i_deg=0.0)
        assert cfg.wavelengths_um == [0.55]

    def test_wavelength_list_passthrough(self):
        cfg = self._base(
            wavelength_um=[0.465, 0.525, 0.630], theta_i_deg=0.0
        )
        assert cfg.wavelengths_um == [0.465, 0.525, 0.630]

    def test_wavelength_scalar_property_returns_first_of_list(self):
        cfg = self._base(wavelength_um=[0.465, 0.525], theta_i_deg=0.0)
        assert cfg.wavelength_um == pytest.approx(0.465)

    # ── theta_i_list_deg ────────────────────────────────────────────────

    def test_theta_scalar_normalized_to_list(self):
        cfg = self._base(wavelength_um=0.55, theta_i_deg=20.0)
        assert cfg.theta_i_list_deg == [20.0]

    def test_theta_list_passthrough(self):
        cfg = self._base(wavelength_um=0.55, theta_i_deg=[0, 20, 40, 60])
        assert cfg.theta_i_list_deg == [0.0, 20.0, 40.0, 60.0]

    def test_theta_scalar_property_returns_first_of_list(self):
        cfg = self._base(wavelength_um=0.55, theta_i_deg=[20.0, 40.0])
        assert cfg.theta_i_deg == pytest.approx(20.0)

    # ── modes ────────────────────────────────────────────────────────────

    def test_mode_unset_returns_empty_list(self):
        cfg = self._base(wavelength_um=0.55, theta_i_deg=20.0)
        assert cfg.modes == []

    def test_mode_scalar_string(self):
        cfg = self._base(wavelength_um=0.55, theta_i_deg=20.0, mode="BRDF")
        assert cfg.modes == ["BRDF"]

    def test_mode_list(self):
        cfg = self._base(
            wavelength_um=0.55, theta_i_deg=20.0, mode=["BRDF", "BTDF"]
        )
        assert cfg.modes == ["BRDF", "BTDF"]

    def test_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="BRDF.*BTDF"):
            self._base(wavelength_um=0.55, theta_i_deg=20.0, mode="Invalid")

    def test_mode_invalid_in_list_raises(self):
        with pytest.raises(ValueError, match="BRDF.*BTDF"):
            self._base(
                wavelength_um=0.55, theta_i_deg=20.0, mode=["BRDF", "XYZ"]
            )

    # ── conditions（直積） ──────────────────────────────────────────────

    def test_single_condition(self):
        cfg = self._base(wavelength_um=0.55, theta_i_deg=0.0)
        conds = cfg.conditions
        assert len(conds) == 1
        assert conds[0]["wavelength_um"] == pytest.approx(0.55)
        assert conds[0]["theta_i_deg"] == pytest.approx(0.0)
        assert conds[0]["mode"] == "BRDF"

    def test_legacy_btdf_auto_detect(self):
        """mode 未指定で theta_i > 90° → BTDF (effective angle)。"""
        cfg = self._base(wavelength_um=0.55, theta_i_deg=150.0)
        conds = cfg.conditions
        assert len(conds) == 1
        assert conds[0]["mode"] == "BTDF"
        assert conds[0]["theta_i_deg"] == pytest.approx(30.0)  # |180 - 150|

    def test_mixed_brdf_btdf_legacy(self):
        """mode 未指定で theta_i list に BRDF と BTDF を混在。"""
        cfg = self._base(wavelength_um=0.55, theta_i_deg=[20.0, 150.0])
        conds = cfg.conditions
        assert len(conds) == 2
        assert conds[0]["mode"] == "BRDF"
        assert conds[0]["theta_i_deg"] == pytest.approx(20.0)
        assert conds[1]["mode"] == "BTDF"
        assert conds[1]["theta_i_deg"] == pytest.approx(30.0)

    def test_explicit_mode_cartesian_product(self):
        """新書式: theta_i × mode の直積展開。"""
        cfg = self._base(
            wavelength_um=0.55,
            theta_i_deg=[0, 20],
            mode=["BRDF", "BTDF"],
        )
        conds = cfg.conditions
        assert len(conds) == 4  # 2 theta × 2 mode
        key_set = {(c["theta_i_deg"], c["mode"]) for c in conds}
        assert (0.0, "BRDF") in key_set
        assert (0.0, "BTDF") in key_set
        assert (20.0, "BRDF") in key_set
        assert (20.0, "BTDF") in key_set

    def test_full_grid_24_conditions(self):
        """LightTools サンプル相当: 3λ × 4 AOI × 2 mode = 24 条件。"""
        cfg = self._base(
            wavelength_um=[0.465, 0.525, 0.630],
            theta_i_deg=[0, 20, 40, 60],
            mode=["BRDF", "BTDF"],
        )
        conds = cfg.conditions
        assert len(conds) == 24

    def test_conditions_carry_optical_params(self):
        cfg = self._base(
            wavelength_um=0.55, theta_i_deg=20.0,
            n1=1.0, n2=1.5, polarization="S", phi_i_deg=10.0,
        )
        c = cfg.conditions[0]
        assert c["n1"] == pytest.approx(1.0)
        assert c["n2"] == pytest.approx(1.5)
        assert c["polarization"] == "S"
        assert c["phi_i_deg"] == pytest.approx(10.0)

    def test_theta_90_raises_in_list(self):
        with pytest.raises(ValueError):
            self._base(wavelength_um=0.55, theta_i_deg=[30, 90])


class TestMeasuredBsdfConfig:
    """measured_bsdf セクションの config 読み込み。"""

    def _base(self, measured_bsdf=None):
        cfg = {
            "simulation": {
                "wavelength_um": 0.55, "theta_i_deg": 0.0,
                "phi_i_deg": 0.0, "n1": 1.0, "n2": 1.5,
                "polarization": "Unpolarized",
            },
            "surface": {"model": "RandomRoughSurface"},
            "error_metrics": {"bsdf_floor": 1e-6},
        }
        if measured_bsdf is not None:
            cfg["measured_bsdf"] = measured_bsdf
        return BSDFConfig(cfg)

    def test_default_absent(self):
        cfg = self._base()
        assert cfg.measured_bsdf == {}
        assert cfg.measured_bsdf_path is None
        assert cfg.match_measured is False

    def test_path_only(self):
        cfg = self._base(measured_bsdf={"path": "data.bsdf"})
        assert cfg.measured_bsdf_path == "data.bsdf"
        assert cfg.match_measured is False

    def test_match_measured_true(self):
        cfg = self._base(measured_bsdf={
            "path": "data.bsdf", "match_measured": True
        })
        assert cfg.match_measured is True

    def test_tolerance_defaults(self):
        cfg = self._base(measured_bsdf={"path": "data.bsdf"})
        assert cfg.match_tolerance_deg == pytest.approx(1.0)
        assert cfg.match_tolerance_nm == pytest.approx(5.0)

    def test_tolerance_override(self):
        cfg = self._base(measured_bsdf={
            "path": "data.bsdf",
            "tolerance_deg": 2.5,
            "tolerance_nm": 10.0,
        })
        assert cfg.match_tolerance_deg == pytest.approx(2.5)
        assert cfg.match_tolerance_nm == pytest.approx(10.0)


class TestFFTMode:
    """config.fft.mode の読み込みとバリデーション。"""

    def _base(self, **overrides):
        cfg = {**_MINIMAL_CFG, **overrides}
        return BSDFConfig(cfg)

    def test_default_is_tilt(self):
        assert self._base().fft_mode == "tilt"

    def test_explicit_tilt(self):
        assert self._base(fft={"mode": "tilt"}).fft_mode == "tilt"

    def test_output_shift(self):
        assert self._base(fft={"mode": "output_shift"}).fft_mode == "output_shift"

    def test_zero(self):
        assert self._base(fft={"mode": "zero"}).fft_mode == "zero"

    def test_invalid_raises(self):
        cfg = self._base(fft={"mode": "unknown"})
        with pytest.raises(ValueError, match="fft.mode"):
            _ = cfg.fft_mode


class TestSecondaryXUnit:
    """config.visualization.secondary_x_unit の読み込みとバリデーション。"""

    def _base(self, **overrides):
        cfg = {**_MINIMAL_CFG, **overrides}
        return BSDFConfig(cfg)

    def test_default_is_lambda_scale(self):
        assert self._base().secondary_x_unit == "lambda_scale"

    @pytest.mark.parametrize(
        "unit", ["lambda_scale", "u", "f", "k_x", "theta_s"]
    )
    def test_valid_units(self, unit):
        cfg = self._base(visualization={"secondary_x_unit": unit})
        assert cfg.secondary_x_unit == unit

    def test_invalid_raises(self):
        cfg = self._base(visualization={"secondary_x_unit": "bogus"})
        with pytest.raises(ValueError, match="secondary_x_unit"):
            _ = cfg.secondary_x_unit
