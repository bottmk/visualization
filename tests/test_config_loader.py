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
                    "illumination": {"preset": "green"},
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

    def test_green_illumination_wavelength(self):
        cfg = self._make_cfg_with_sparkle("smartphone")
        sparkle = cfg.metrics["sparkle"]
        assert sparkle["illumination"]["wavelengths_um"] == [0.55]

    def test_unknown_preset_raises(self):
        raw = {
            **_MINIMAL_CFG,
            "metrics": {
                "sparkle": {
                    "viewing": {"preset": "unknown_device"},
                    "display": {"preset": "fhd_smartphone"},
                    "illumination": {"preset": "green"},
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
                    "illumination": {"preset": "green"},
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
