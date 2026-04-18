"""YAML設定ファイルの読み込み・バリデーション・プリセット解決。"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml


def _as_list(value: Any) -> list:
    """スカラまたは list を list に正規化する。None は空 list。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# ── プリセット定義 ────────────────────────────────────────────────────────────

_VIEWING_PRESETS: dict[str, dict[str, float]] = {
    "smartphone": {"distance_mm": 300.0, "pupil_diameter_mm": 3.0},
    "tablet":     {"distance_mm": 350.0, "pupil_diameter_mm": 3.0},
    "monitor":    {"distance_mm": 600.0, "pupil_diameter_mm": 3.0},
}

_DISPLAY_PRESETS: dict[str, dict[str, Any]] = {
    "fhd_smartphone": {"pixel_pitch_mm": 0.062, "subpixel_layout": "rgb_stripe"},
    "qhd_monitor":    {"pixel_pitch_mm": 0.124, "subpixel_layout": "rgb_stripe"},
    "4k_monitor":     {"pixel_pitch_mm": 0.160, "subpixel_layout": "rgb_stripe"},
}

_ADDING_DOUBLING_PRESETS: dict[str, dict[str, int]] = {
    "fast":     {"n_theta": 32,  "m_phi": 8},
    "standard": {"n_theta": 128, "m_phi": 18},
    "high":     {"n_theta": 256, "m_phi": 36},
}


def _resolve_preset(
    section: dict[str, Any],
    presets: dict[str, dict[str, Any]],
    preset_key: str = "preset",
) -> dict[str, Any]:
    """プリセットと個別数値を解決する。

    数値が明示されている場合（null でない）は数値を優先する。
    数値が null の場合はプリセット値を使用する。
    プリセットも未指定の場合はエラーとする。

    Args:
        section: YAML のセクション辞書
        presets: プリセット定義辞書
        preset_key: プリセット名のキー

    Returns:
        解決済みのパラメータ辞書
    """
    preset_name = section.get(preset_key)
    result: dict[str, Any] = {}

    if preset_name and preset_name != "custom":
        if preset_name not in presets:
            raise ValueError(f"未知のプリセット: '{preset_name}'。有効なプリセット: {list(presets.keys())}")
        result.update(presets[preset_name])

    # 個別数値でオーバーライド（null 以外の場合）
    for key, value in section.items():
        if key == preset_key:
            continue
        if value is not None:
            result[key] = value

    # すべての値が未設定の場合はエラー
    if not result:
        raise ValueError(
            f"プリセットまたは個別数値を指定する必要がある。"
            f"プリセット key='{preset_key}' が未指定かつ全値が null。"
        )

    return result


class BSDFConfig:
    """YAML設定ファイルを読み込み、プリセット解決とバリデーションを行うクラス。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self._raw = config
        self._resolved: dict[str, Any] = {}
        self._resolve()
        self._validate()

    @classmethod
    def from_file(cls, path: str | Path) -> "BSDFConfig":
        """YAMLファイルから設定を読み込む。

        Args:
            path: 設定ファイルのパス

        Returns:
            BSDFConfig インスタンス
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"設定ファイルが見つからない: {path}")
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw)

    def _resolve(self) -> None:
        """プリセットを解決し、_resolved に格納する。"""
        self._resolved = dict(self._raw)

        # Adding-Doubling プリセット解決
        if ad := self._raw.get("adding_doubling"):
            precision = ad.get("precision", "standard")
            resolved_ad = dict(_ADDING_DOUBLING_PRESETS.get(precision, _ADDING_DOUBLING_PRESETS["standard"]))
            if ad.get("n_theta") is not None:
                resolved_ad["n_theta"] = ad["n_theta"]
            if ad.get("m_phi") is not None:
                resolved_ad["m_phi"] = ad["m_phi"]
            self._resolved["adding_doubling"] = {**ad, **resolved_ad}

        # Sparkle プリセット解決
        if metrics := self._raw.get("metrics"):
            if sparkle := metrics.get("sparkle"):
                resolved_sparkle = dict(sparkle)

                if viewing := sparkle.get("viewing"):
                    resolved_sparkle["viewing"] = _resolve_preset(viewing, _VIEWING_PRESETS)

                if display := sparkle.get("display"):
                    resolved_sparkle["display"] = _resolve_preset(display, _DISPLAY_PRESETS)

                # illumination プリセットは simulate ループが波長ごとに独立に
                # 計算するため使われない。指定されていても無視する（後方互換のため
                # resolved_sparkle から削除）。
                resolved_sparkle.pop("illumination", None)

                self._resolved.setdefault("metrics", {})["sparkle"] = resolved_sparkle

    def _validate(self) -> None:
        """必須フィールドと値の範囲を検証する。"""
        sim = self._resolved.get("simulation", {})

        # theta_i_deg（スカラ / list 両対応）: 90° はエラー
        theta_raw = sim.get("theta_i_deg", 0.0)
        for t in _as_list(theta_raw):
            if abs(float(t) - 90.0) < 1e-6:
                raise ValueError("theta_i_deg = 90° は未定義。BRDF(<90°) または BTDF(>90°) を指定すること。")

        # polarization
        pol = sim.get("polarization", "Unpolarized")
        if pol not in ("S", "P", "Unpolarized"):
            raise ValueError(f"polarization は 'S' / 'P' / 'Unpolarized' のいずれかでなければならない。値={pol}")

        # mode（指定時）
        mode_raw = sim.get("mode")
        if mode_raw is not None:
            for m in _as_list(mode_raw):
                if m not in ("BRDF", "BTDF"):
                    raise ValueError(
                        f"simulation.mode は 'BRDF' / 'BTDF' のみ。値={m}"
                    )

        # bsdf_floor
        floor = self._resolved.get("error_metrics", {}).get("bsdf_floor", 1e-6)
        if floor <= 0:
            raise ValueError(f"bsdf_floor は正の値でなければならない。値={floor}")

    def get(self, key: str, default: Any = None) -> Any:
        """解決済み設定から値を取得する。"""
        return self._resolved.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._resolved[key]

    # ── ショートカットプロパティ ──────────────────────────────────────────

    @property
    def simulation(self) -> dict[str, Any]:
        return self._resolved.get("simulation", {})

    @property
    def surface(self) -> dict[str, Any]:
        return self._resolved.get("surface", {})

    @property
    def adding_doubling(self) -> dict[str, Any]:
        return self._resolved.get("adding_doubling", {})

    @property
    def error_metrics(self) -> dict[str, Any]:
        return self._resolved.get("error_metrics", {"bsdf_floor": 1e-6})

    @property
    def bsdf_floor(self) -> float:
        return float(self.error_metrics.get("bsdf_floor", 1e-6))

    @property
    def metrics(self) -> dict[str, Any]:
        return self._resolved.get("metrics", {})

    @property
    def optuna(self) -> dict[str, Any]:
        return self._resolved.get("optuna", {})

    @property
    def mlflow(self) -> dict[str, Any]:
        return self._resolved.get("mlflow", {})

    @property
    def dynamicmap(self) -> dict[str, Any]:
        return self._resolved.get("dynamicmap", {})

    @property
    def wavelength_um(self) -> float:
        """主波長 [μm]（list の場合は先頭要素。1 条件 API 互換用）。"""
        wls = self.wavelengths_um
        return wls[0] if wls else 0.55

    @property
    def theta_i_deg(self) -> float:
        """主入射角 [deg]（list の場合は先頭要素。1 条件 API 互換用）。"""
        thetas = self.theta_i_list_deg
        return thetas[0] if thetas else 0.0

    @property
    def phi_i_deg(self) -> float:
        return float(self.simulation.get("phi_i_deg", 0.0))

    @property
    def n1(self) -> float:
        return float(self.simulation.get("n1", 1.0))

    @property
    def n2(self) -> float:
        return float(self.simulation.get("n2", 1.5))

    @property
    def polarization(self) -> str:
        return str(self.simulation.get("polarization", "Unpolarized"))

    @property
    def is_btdf(self) -> bool:
        """BTDF モード（theta_i > 90° または主条件が BTDF）かどうか。"""
        conds = self.conditions
        if conds:
            return conds[0]["mode"] == "BTDF"
        return False

    @property
    def theta_i_effective_deg(self) -> float:
        """BTDF モード時に表面側座標系に換算した有効入射角（主条件）。"""
        conds = self.conditions
        if conds:
            return float(conds[0]["theta_i_deg"])
        return 0.0

    # ── 多条件サポート（案 2-B: スカラ/list 両対応）──────────────────────

    @property
    def wavelengths_um(self) -> list[float]:
        """正規化された波長リスト [μm]（スカラも list も受理）。"""
        raw = self.simulation.get("wavelength_um", 0.55)
        return [float(w) for w in _as_list(raw)]

    @property
    def theta_i_list_deg(self) -> list[float]:
        """正規化された入射角リスト [deg]（スカラも list も受理）。"""
        raw = self.simulation.get("theta_i_deg", 0.0)
        return [float(t) for t in _as_list(raw)]

    @property
    def modes(self) -> list[str]:
        """simulation.mode の正規化リスト。未指定時は空 list（旧互換判定を使用）。"""
        raw = self.simulation.get("mode")
        return [str(m) for m in _as_list(raw)]

    @property
    def conditions(self) -> list[dict[str, Any]]:
        """正規化された光学条件リスト。各要素は 1 シミュレーション条件の dict。

        生成規則:
          - mode 未指定（旧互換）: theta_i_deg > 90° → BTDF に自動判定、
            theta_i_effective = |180 - theta_i|
          - mode 指定あり（新書式）: theta_i_deg × mode の直積
          - wavelength_um 軸は常に直積展開

        Returns:
            dict リスト。各要素のキー: wavelength_um, theta_i_deg, mode,
            phi_i_deg, n1, n2, polarization
        """
        wls = self.wavelengths_um or [0.55]
        thetas_raw = self.theta_i_list_deg or [0.0]
        modes_raw = self.modes

        if not modes_raw:
            # 旧互換: theta_i > 90° → BTDF、有効角 = |180 - theta|
            theta_mode_pairs: list[tuple[float, str]] = []
            for t in thetas_raw:
                if t > 90.0:
                    theta_mode_pairs.append((abs(180.0 - t), "BTDF"))
                else:
                    theta_mode_pairs.append((t, "BRDF"))
        else:
            # 新書式: theta_i × mode 直積
            theta_mode_pairs = [(t, m) for t, m in itertools.product(thetas_raw, modes_raw)]

        conditions: list[dict[str, Any]] = []
        for wl in wls:
            for theta_i, mode in theta_mode_pairs:
                conditions.append({
                    "wavelength_um": wl,
                    "theta_i_deg": theta_i,
                    "mode": mode,
                    "phi_i_deg": self.phi_i_deg,
                    "n1": self.n1,
                    "n2": self.n2,
                    "polarization": self.polarization,
                })
        return conditions

    # ── measured_bsdf セクション ─────────────────────────────────────────

    @property
    def measured_bsdf(self) -> dict[str, Any]:
        return self._resolved.get("measured_bsdf", {})

    @property
    def measured_bsdf_path(self) -> str | None:
        path = self.measured_bsdf.get("path")
        return str(path) if path else None

    @property
    def match_measured(self) -> bool:
        """True: 実測ファイル内の条件を sim 条件として自動採用する。"""
        return bool(self.measured_bsdf.get("match_measured", False))

    @property
    def match_tolerance_deg(self) -> float:
        """sim 条件と実測ブロックのマッチング許容角 [deg]。デフォルト 1.0°。"""
        return float(self.measured_bsdf.get("tolerance_deg", 1.0))

    @property
    def match_tolerance_nm(self) -> float:
        """sim 条件と実測ブロックのマッチング許容波長 [nm]。デフォルト 5.0nm。"""
        return float(self.measured_bsdf.get("tolerance_nm", 5.0))

    # ── FFT 法のオプション ─────────────────────────────────────────────────

    @property
    def fft(self) -> dict[str, Any]:
        return self._resolved.get("fft", {})

    @property
    def fft_mode(self) -> str:
        """FFT 法の計算モード。'tilt'（既定）/ 'output_shift' / 'zero'。

        詳細は docs/fft_bsdf_math.md の「3 つの fft_mode オプション」参照。
        """
        mode = str(self.fft.get("mode", "tilt"))
        valid = ("tilt", "output_shift", "zero")
        if mode not in valid:
            raise ValueError(
                f"config.fft.mode は {valid} のいずれかでなければならない。"
                f"値={mode!r}"
            )
        return mode

    @property
    def fft_apply_fresnel(self) -> bool:
        """FFT 法出力に θ_i のフレネル反射/透過率を後掛けするか。

        True の場合、BRDF では R(θ_i)=(|r_s|²+|r_p|²)/2、BTDF では
        T(θ_i)=(n2·cos θ_t)/(n1·cos θ_i)·(|t_s|²+|t_p|²)/2 を全 BSDF 値に掛ける。
        θ_s 依存性は含まれないヒューリスティック補正なので、厳密には PSD 法を
        使うのが本筋。
        """
        return bool(self.fft.get("apply_fresnel", False))

    # ── 可視化オプション ───────────────────────────────────────────────────

    @property
    def visualization(self) -> dict[str, Any]:
        return self._resolved.get("visualization", {})

    @property
    def secondary_x_unit(self) -> str:
        """BSDF 1D プロットの副軸（上段 X 軸）ユニット。

        有効値:
            - 'lambda_scale'（既定）: 構造スケール Λ = λ/sin θ_s [μm]
            - 'u': 方向余弦 sin θ_s
            - 'f': 空間周波数 sin θ_s / λ [μm⁻¹]
            - 'k_x': 横方向波数 2π sin θ_s / λ [rad/μm]
            - 'theta_s': 副軸なし

        詳細は src/bsdf_sim/visualization/secondary_axis.py 参照。
        """
        valid = ("lambda_scale", "u", "f", "k_x", "theta_s")
        unit = str(self.visualization.get("secondary_x_unit", "lambda_scale"))
        if unit not in valid:
            raise ValueError(
                f"config.visualization.secondary_x_unit は {valid} のいずれか。"
                f"値={unit!r}"
            )
        return unit

    @property
    def metric_overlay(self) -> dict[str, Any]:
        """BSDF 2D ヒートマップへの光学指標オーバーレイ設定。

        config.visualization.metric_overlay セクション:
            show_overlay (bool): 全体 on/off（既定 True）
            initially_shown (list[str] | None): 初期表示するキー（None で全表示）
                使えるキー: 'haze' / 'gloss_20' / 'gloss_60' / 'gloss_85' /
                            'doi_nser' / 'doi_comb' / 'doi_astm'
            click_policy (str): 'hide' / 'mute' / 'none'（既定 'hide'）
            legend_position (str): 'right' / 'top_right' / 'bottom' 等（既定 'right'）
        """
        return self.visualization.get("metric_overlay") or {}

    # ── 代表波長（規格準拠の Haze/Gloss/DOI 計算用）────────────────────────

    @property
    def representative_wavelength_um(self) -> float:
        """Haze/Gloss/DOI を計算する代表波長 [μm]。デフォルト 0.555（V(λ) ピーク）。

        JIS K 7136 / ISO 14782 / ASTM D1003（Haze）、ISO 2813 / ASTM D523
        （Gloss）、JIS K 7374 / ASTM E430（DOI）等の規格は CIE 白色光源 + 明所視応答
        V(λ) で規定されている。本実装は V(λ) ピーク波長（555nm）での単波長
        近似を採用する（AG フィルム等、凹凸 >> 波長の構造では近似誤差 1〜5%）。

        `simulation.wavelength_um` が list でも、Haze/Gloss/DOI の値は
        この 1 波長でのみ計算される。メトリクス名は <name>_<method>_<deg>_<mode>
        （例: `haze_fft_0_t`, `gloss_fft_60_r`, `doi_nser_fft_0_t`,
        `doi_comb_fft_0_t`, `doi_astm_fft_20_r`）で、各規格 (θ_i, mode) に
        対応する列が生成される。
        """
        return float(self.metrics.get("representative_wavelength_um", 0.555))
