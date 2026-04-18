"""CLI エントリポイント（4サブコマンド）。

bsdf simulate  --config config.yaml
bsdf optimize  --config config.yaml --trials 100
bsdf visualize --run-id <mlflow_run_id>
bsdf report    --run-ids <id1>,<id2>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import time

import click
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _elapsed(t0: float) -> str:
    """経過時間を秒単位の文字列で返す。"""
    return f"{time.perf_counter() - t0:.2f}s"
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="bsdf")
def cli() -> None:
    """BSDF シミュレーション・最適化 CLI。"""


# ── simulate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="設定ファイルパス（YAML）")
@click.option("--output-dir", "-o", default="outputs", show_default=True, help="出力ディレクトリ")
@click.option("--method", "-m", default="both", type=click.Choice(["fft", "psd", "both"]), show_default=True, help="計算手法")
@click.option("--save-parquet/--no-save-parquet", default=True, show_default=True, help="Parquet 保存の有無")
@click.option("--log-to-mlflow/--no-log-to-mlflow", default=False, show_default=True, help="MLflow への記録")
@click.option("--measured-bsdf", default=None, type=click.Path(exists=True),
              help="BSDF 実測ファイルパス（.bsdf）。指定時は config.measured_bsdf.path を上書き。")
@click.option("--match-measured/--no-match-measured", default=None,
              help="実測ファイル内の光学条件を sim 条件として自動採用するか。省略時は config の値を使用。")
def simulate(
    config: str,
    output_dir: str,
    method: str,
    save_parquet: bool,
    log_to_mlflow: bool,
    measured_bsdf: str | None,
    match_measured: bool | None,
) -> None:
    """BSDF シミュレーションを単体実行する。"""
    from ..io.config_loader import BSDFConfig
    from ..io import get_conditions as _meas_get_conditions, load_bsdf_readers, read_bsdf_file
    from ..io.parquet_schema import build_dataframe, merge_sim_and_measured
    from ..io.parquet_schema import save_parquet as _save_parquet
    from ..metrics.optical import compute_all_optical_metrics
    from ..metrics.surface import compute_all_surface_metrics
    from ..models import create_model_from_config, load_plugins
    from ..optics.fft_bsdf import compute_bsdf_fft
    from ..optics.psd_bsdf import compute_bsdf_psd

    cfg = BSDFConfig.from_file(config)
    load_plugins()

    logger.info(f"設定ファイル読み込み完了: {config}")
    logger.info(f"表面モデル: {cfg.surface.get('model')}")

    # [1] 表面形状モデルの生成
    t0 = time.perf_counter()
    logger.info(f"[1/4] 高さマップ生成中... (grid={cfg.surface.get('grid_size', 4096)})")
    model = create_model_from_config(cfg._resolved)
    hm = model.get_height_map()
    logger.info(f"      完了 ({_elapsed(t0)})  {hm.grid_size}×{hm.grid_size}, pixel={hm.pixel_size_um}μm")

    # [2] 表面形状指標
    t0 = time.perf_counter()
    logger.info("[2/4] 表面形状指標計算中...")
    surface_metrics = compute_all_surface_metrics(hm, verbose=True)
    logger.info(f"      完了 ({_elapsed(t0)})")
    for k, v in surface_metrics.items():
        logger.info(f"        {k} = {v:.6f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    approx_mode = cfg._resolved.get("psd", {}).get("approx_mode", False)

    # ── 実測 BSDF の読み込み（CLI/config どちらからでも）──────────────────
    effective_measured_path = measured_bsdf or cfg.measured_bsdf_path
    effective_match_measured = match_measured if match_measured is not None else cfg.match_measured

    measured_dfs: list = []
    if effective_measured_path:
        logger.info(f"BSDF 実測ファイル読み込み中: {effective_measured_path}")
        load_bsdf_readers()
        measured_dfs = read_bsdf_file(effective_measured_path)
        logger.info(f"      ブロック数: {len(measured_dfs)}")

    # ── 光学条件リストの決定 ──────────────────────────────────────────────
    if effective_match_measured and measured_dfs:
        meas_conds = _meas_get_conditions(measured_dfs)
        conditions = [
            {
                "wavelength_um": c["wavelength_um"],
                "theta_i_deg":   c["theta_i_deg"],
                "mode":          c["mode"],
                "phi_i_deg":     cfg.phi_i_deg,
                "n1":            cfg.n1,
                "n2":            cfg.n2,
                "polarization":  cfg.polarization,
            }
            for c in meas_conds
        ]
        logger.info(
            f"match_measured=true: 実測ファイルから {len(conditions)} 条件を採用"
        )
    else:
        conditions = cfg.conditions

    multi = len(conditions) > 1
    logger.info(
        f"光学条件数: {len(conditions)} "
        f"({len(cfg.wavelengths_um)} λ × {len(cfg.theta_i_list_deg)} θ_i × "
        f"{len(cfg.modes) or 1} mode)"
    )

    # ── BSDF 計算（全条件ループ）────────────────────────────────────────
    step3_label = {"fft": "FFT", "psd": "PSD", "both": "FFT + PSD"}[method]
    logger.info(f"[3/4] {step3_label} 計算中... ({len(conditions)} 条件)")

    all_dfs: list = []
    # 条件ごとの (u, v, bsdf) を手法別に保持（光学指標計算で使う）
    method_bsdf_by_cond: dict[tuple, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    # Adding-Doubling 用 primary（最初の条件の最初の手法）
    u_primary: np.ndarray | None = None
    v_primary: np.ndarray | None = None
    bsdf_primary: np.ndarray | None = None

    for i_cond, cond in enumerate(conditions):
        wl_um = cond["wavelength_um"]
        theta_i = cond["theta_i_deg"]
        mode = cond["mode"]
        is_btdf = mode == "BTDF"
        cond_key = (wl_um, theta_i, mode)
        method_bsdf_by_cond[cond_key] = {}

        logger.info(
            f"      [{i_cond + 1}/{len(conditions)}] "
            f"λ={wl_um * 1000:.0f}nm, θ_i={theta_i:.1f}°, {mode}"
        )

        if method in ("fft", "both"):
            t0 = time.perf_counter()
            u, v, bsdf_fft = compute_bsdf_fft(
                height_map=hm,
                wavelength_um=wl_um,
                theta_i_deg=theta_i,
                phi_i_deg=cond["phi_i_deg"],
                n1=cond["n1"],
                n2=cond["n2"],
                polarization=cond["polarization"],
                is_btdf=is_btdf,
                fft_mode=cfg.fft_mode,
                apply_fresnel=cfg.fft_apply_fresnel,
            )
            method_bsdf_by_cond[cond_key]["fft"] = (u, v, bsdf_fft)
            if u_primary is None:
                u_primary, v_primary, bsdf_primary = u, v, bsdf_fft
            all_dfs.append(build_dataframe(
                u, v, bsdf_fft, "FFT",
                theta_i, cond["phi_i_deg"], wl_um, cond["polarization"],
                is_btdf=is_btdf,
            ))
            logger.info(
                f"          FFT 完了 ({_elapsed(t0)}, mode={cfg.fft_mode}"
                f"{', +Fresnel' if cfg.fft_apply_fresnel else ''})"
            )

        if method in ("psd", "both"):
            t0 = time.perf_counter()
            u, v, bsdf_psd = compute_bsdf_psd(
                height_map=hm,
                wavelength_um=wl_um,
                theta_i_deg=theta_i,
                phi_i_deg=cond["phi_i_deg"],
                n1=cond["n1"],
                n2=cond["n2"],
                polarization=cond["polarization"],
                is_btdf=is_btdf,
                approx_mode=approx_mode,
            )
            method_bsdf_by_cond[cond_key]["psd"] = (u, v, bsdf_psd)
            if u_primary is None:
                u_primary, v_primary, bsdf_primary = u, v, bsdf_psd
            all_dfs.append(build_dataframe(
                u, v, bsdf_psd, "PSD",
                theta_i, cond["phi_i_deg"], wl_um, cond["polarization"],
                is_btdf=is_btdf,
            ))
            logger.info(f"          PSD 完了 ({_elapsed(t0)})")

    # ── 代表波長での Haze/Gloss/DOI 用 BSDF を確保 ────────────────────────
    # 規格（JIS K 7136 / ISO 2813 / JIS K 7374 / ASTM E430 等）は CIE 白色光源 + V(λ)
    # で規定されており、本実装は V(λ) ピーク付近の単波長近似を採用する。
    # 代表波長は metrics.representative_wavelength_um（デフォルト 0.555μm）。
    #
    # 各指標の規格条件 (θ_i, mode) に必要な BSDF 条件を列挙し、sim リストに
    # なければ自動追加する（Q1.(a) 方針）。
    rep_wl_um = cfg.representative_wavelength_um
    primary_cond = conditions[0]

    # 各指標の規格条件を集める: (wl_um, theta_deg, mode) の set
    required_standard_conds: set[tuple[float, float, str]] = set()
    m = cfg.metrics
    if m.get("haze") is not None and m["haze"].get("enabled", True):
        required_standard_conds.add((rep_wl_um, 0.0, "BTDF"))
    if m.get("gloss") is not None and m["gloss"].get("enabled", True):
        for ang in m["gloss"].get("enabled_angles", [20, 60, 85]):
            required_standard_conds.add((rep_wl_um, float(ang), "BRDF"))
    if m.get("doi_nser") is not None and m["doi_nser"].get("enabled", True):
        required_standard_conds.add((rep_wl_um, 0.0, "BTDF"))
    if m.get("doi_comb") is not None and m["doi_comb"].get("enabled", True):
        for mc in m["doi_comb"].get("enabled_modes", ["t", "r"]):
            mode_std = "BTDF" if mc == "t" else "BRDF"
            required_standard_conds.add((rep_wl_um, 0.0, mode_std))
    if m.get("doi_astm") is not None and m["doi_astm"].get("enabled", True):
        for ang in m["doi_astm"].get("enabled_angles", [20, 30]):
            required_standard_conds.add((rep_wl_um, float(ang), "BRDF"))

    def _find_existing(wl: float, theta: float, mode_s: str) -> tuple | None:
        for ck in method_bsdf_by_cond:
            wl_k, t_k, m_k = ck
            if (
                abs(wl_k - wl) < 1e-6
                and abs(t_k - theta) < 1e-6
                and m_k == mode_s
            ):
                return ck
        return None

    # 規格条件のうち既存に無いものを追加 sim
    for (wl_std, theta_std, mode_std) in sorted(required_standard_conds):
        if _find_existing(wl_std, theta_std, mode_std) is not None:
            continue
        is_btdf_std = mode_std == "BTDF"
        logger.info(
            f"規格条件の追加 sim を実行中: "
            f"λ={wl_std * 1000:.0f}nm, θ_i={theta_std:.1f}°, {mode_std}"
        )
        std_key = (wl_std, theta_std, mode_std)
        method_bsdf_by_cond[std_key] = {}
        if method in ("fft", "both"):
            t0 = time.perf_counter()
            u_r, v_r, bsdf_r = compute_bsdf_fft(
                height_map=hm,
                wavelength_um=wl_std,
                theta_i_deg=theta_std,
                phi_i_deg=primary_cond["phi_i_deg"],
                n1=primary_cond["n1"],
                n2=primary_cond["n2"],
                polarization=primary_cond["polarization"],
                is_btdf=is_btdf_std,
                fft_mode=cfg.fft_mode,
                apply_fresnel=cfg.fft_apply_fresnel,
            )
            method_bsdf_by_cond[std_key]["fft"] = (u_r, v_r, bsdf_r)
            if u_primary is None:
                u_primary, v_primary, bsdf_primary = u_r, v_r, bsdf_r
            all_dfs.append(build_dataframe(
                u_r, v_r, bsdf_r, "FFT",
                theta_std, primary_cond["phi_i_deg"], wl_std,
                primary_cond["polarization"], is_btdf=is_btdf_std,
            ))
            logger.info(f"          FFT 完了 ({_elapsed(t0)})")
        if method in ("psd", "both"):
            t0 = time.perf_counter()
            u_r, v_r, bsdf_r = compute_bsdf_psd(
                height_map=hm,
                wavelength_um=wl_std,
                theta_i_deg=theta_std,
                phi_i_deg=primary_cond["phi_i_deg"],
                n1=primary_cond["n1"],
                n2=primary_cond["n2"],
                polarization=primary_cond["polarization"],
                is_btdf=is_btdf_std,
                approx_mode=approx_mode,
            )
            method_bsdf_by_cond[std_key]["psd"] = (u_r, v_r, bsdf_r)
            if u_primary is None:
                u_primary, v_primary, bsdf_primary = u_r, v_r, bsdf_r
            all_dfs.append(build_dataframe(
                u_r, v_r, bsdf_r, "PSD",
                theta_std, primary_cond["phi_i_deg"], wl_std,
                primary_cond["polarization"], is_btdf=is_btdf_std,
            ))
            logger.info(f"          PSD 完了 ({_elapsed(t0)})")

    # Adding-Doubling 多層合成（主条件のみ）
    if cfg.adding_doubling.get("enabled", False) and bsdf_primary is not None:
        if multi:
            logger.warning(
                "Adding-Doubling は多条件実行の最初の条件にのみ適用される。"
            )
        logger.info("Adding-Doubling 多層合成を実行中...")
        from ..optics.multilayer import MultiLayerBSDF
        ad_cfg = cfg.adding_doubling
        ml_bsdf = MultiLayerBSDF(
            precision=ad_cfg.get("precision", "standard"),
            n_theta=ad_cfg.get("n_theta"),
            m_phi=ad_cfg.get("m_phi"),
        )
        for layer in ad_cfg.get("layers", []):
            layer_type = layer.get("type")
            if layer_type == "surface":
                ml_bsdf.add_surface_layer(bsdf_primary, u_primary, v_primary)
            elif layer_type == "bulk":
                ml_bsdf.add_bulk_layer(
                    g=float(layer.get("hg_g", 0.8)),
                    scattering_coeff_um=float(layer.get("scattering_coeff", 0.1)),
                    thickness_um=float(layer.get("thickness_um", 100.0)),
                )
        u_ml, v_ml, bsdf_ml = ml_bsdf.to_bsdf_2d(u_primary, v_primary)
        # 主条件の値を使って DataFrame を作成
        primary = conditions[0]
        all_dfs.append(build_dataframe(
            u_ml, v_ml, bsdf_ml, "MultiLayer",
            primary["theta_i_deg"], primary["phi_i_deg"],
            primary["wavelength_um"], primary["polarization"],
            is_btdf=(primary["mode"] == "BTDF"),
        ))
        method_bsdf_by_cond[(primary["wavelength_um"], primary["theta_i_deg"], primary["mode"])]["ml"] = (u_ml, v_ml, bsdf_ml)
        u_primary, v_primary, bsdf_primary = u_ml, v_ml, bsdf_ml
        logger.info("Adding-Doubling 完了。")

    if all_dfs and bsdf_primary is not None:
        import pandas as pd
        df_combined = pd.concat(all_dfs, ignore_index=True)

        # [4] 光学指標計算
        # 新命名規則: <name>_<method>_<deg>_<mode> (単波長) / <name>_<method>_<nm>_<deg>_<mode> (波長依存)
        # 各指標は規格 (θ_i, mode) に一致する条件でのみ計算される。
        # compute_at_sim_angles=true のときは sim 全条件で斜入射値も計算する。
        enabled_metrics = [
            k for k in ("haze", "gloss", "doi_nser", "doi_comb", "doi_astm", "sparkle")
            if cfg.metrics.get(k) is not None and cfg.metrics[k].get("enabled", True)
        ]
        logger.info(
            f"[4/4] 光学指標計算中... ({', '.join(enabled_metrics) or 'なし'})"
        )
        t0 = time.perf_counter()
        all_optical_metrics: dict[str, float] = {}

        compute_at_sim_angles = bool(cfg.metrics.get("compute_at_sim_angles", False))

        for cond_key, method_data in method_bsdf_by_cond.items():
            wl_um, theta_i, mode = cond_key
            wl_nm = int(round(wl_um * 1000))
            is_rep_wl = abs(wl_um - rep_wl_um) < 1e-6
            is_std_cond = cond_key in required_standard_conds

            for method_key, (u_m, v_m, bsdf_m) in method_data.items():
                # 単波長メトリクス (Haze/Gloss/DOI): 代表波長のみ
                if is_rep_wl and (is_std_cond or compute_at_sim_angles):
                    metrics_std = compute_all_optical_metrics(
                        u_grid=u_m, v_grid=v_m, bsdf=bsdf_m,
                        theta_i_deg=theta_i, mode=mode,
                        n1=primary_cond["n1"], n2=primary_cond["n2"],
                        method_name=method_key, wavelength_nm=wl_nm,
                        config=cfg.metrics, bsdf_floor=cfg.bsdf_floor,
                        standards_only=True,
                        allow_oblique=(not is_std_cond),
                    )
                    all_optical_metrics.update(metrics_std)

                # 波長依存メトリクス (Sparkle): 全条件
                metrics_sp = compute_all_optical_metrics(
                    u_grid=u_m, v_grid=v_m, bsdf=bsdf_m,
                    theta_i_deg=theta_i, mode=mode,
                    n1=primary_cond["n1"], n2=primary_cond["n2"],
                    method_name=method_key, wavelength_nm=wl_nm,
                    config=cfg.metrics, bsdf_floor=cfg.bsdf_floor,
                    sparkle_only=True,
                )
                all_optical_metrics.update(metrics_sp)

        logger.info(f"      完了 ({_elapsed(t0)})")
        # 代表波長のみの単波長メトリクスを個別表示
        for k in sorted(all_optical_metrics):
            if any(
                k.startswith(f"{name}_")
                for name in ("haze", "gloss", "doi_nser", "doi_comb", "doi_astm")
            ):
                logger.info(f"        {k} = {all_optical_metrics[k]:.6f}")

        # 実測データとの結合＋Log-RMSE（条件ごと）
        if measured_dfs:
            meas_df_concat = pd.concat(measured_dfs, ignore_index=True)
            df_combined = merge_sim_and_measured(
                sim_df=df_combined,
                meas_df=meas_df_concat,
                bsdf_floor=cfg.bsdf_floor,
                tolerance_deg=cfg.match_tolerance_deg,
                tolerance_nm=cfg.match_tolerance_nm,
            )
            # Log-RMSE を all_optical_metrics にも記録
            sim_keys = df_combined[
                df_combined["method"] != "measured"
            ][["method", "wavelength_um", "theta_i_deg", "mode"]].drop_duplicates()
            for _, row in sim_keys.iterrows():
                mask = (
                    (df_combined["method"].values == row["method"])
                    & (np.abs(df_combined["wavelength_um"].values - row["wavelength_um"]) < 1e-6)
                    & (np.abs(df_combined["theta_i_deg"].values - row["theta_i_deg"]) < 1e-6)
                    & (df_combined["mode"].values == row["mode"])
                )
                rmse_vals = df_combined.loc[mask, "log_rmse"].values
                if len(rmse_vals) == 0:
                    continue
                rmse = float(rmse_vals[0])
                if not np.isnan(rmse):
                    method_key = str(row["method"]).lower()
                    wl_nm = int(round(row["wavelength_um"] * 1000))
                    theta_int = int(round(row["theta_i_deg"]))
                    mode_char = "r" if str(row["mode"]).upper() == "BRDF" else "t"
                    key = f"log_rmse_{method_key}_{wl_nm}_{theta_int}_{mode_char}"
                    all_optical_metrics[key] = rmse
            logger.info(
                f"実測データ結合完了: {len(measured_dfs)} ブロック、"
                f"log_rmse を {sum(1 for k in all_optical_metrics if k.startswith('log_rmse'))} 条件で計算"
            )

        # Parquet 保存
        if save_parquet:
            parquet_path = out_path / "bsdf_data.parquet"
            _save_parquet(df_combined, parquet_path)
            logger.info(f"Parquet 保存: {parquet_path}")

        # MLflow 記録
        if log_to_mlflow:
            import tempfile as _tmpmod
            from ..optimization.mlflow_logger import RawDataLogger
            from ..visualization.holoviews_plots import save_heightmap_png, save_bsdf_2d_png

            mlflow_cfg = cfg.mlflow
            ml_logger = RawDataLogger(tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns"))
            surface_cfg = cfg.surface
            model_name = surface_cfg.get("model", "")
            params: dict = {}
            if model_name == "RandomRoughSurface":
                rr = surface_cfg.get("random_rough", {})
                params = {
                    "rq_um": rr.get("rq_um"),
                    "lc_um": rr.get("lc_um"),
                    "fractal_dim": rr.get("fractal_dim"),
                }
            all_metrics = {**surface_metrics, **all_optical_metrics}

            # PNG / HTML を一時ディレクトリで生成してから MLflow にアップロード
            with _tmpmod.TemporaryDirectory() as _tmpdir:
                _td = Path(_tmpdir)
                _plot_paths = []

                # 表面形状 PNG（MLflow UI でインライン表示）
                _sq_nm = surface_metrics.get("sq_um", 0.0) * 1000
                _sa_nm = surface_metrics.get("sa_um", 0.0) * 1000
                _surf_title = f"{model_name}  Sq={_sq_nm:.2f}nm  Sa={_sa_nm:.2f}nm"
                _surf_png = _td / "surface.png"
                save_heightmap_png(hm, _surf_png, title=_surf_title, unit="nm")
                _plot_paths.append(_surf_png)

                # 2D BSDF PNG（条件 × 手法別）
                for _cond_key, _method_data in method_bsdf_by_cond.items():
                    _wl_um, _theta_i, _mode = _cond_key
                    _wl_nm = int(round(_wl_um * 1000))
                    _theta_int = int(round(_theta_i))
                    _mode_char = "r" if _mode.upper() == "BRDF" else "t"
                    _cond_tag = (
                        ""
                        if not multi
                        else f"_{_wl_nm}_{_theta_int}_{_mode_char}"
                    )
                    for _mkey, (_u, _v, _b) in _method_data.items():
                        _bsdf_png = _td / f"bsdf_2d_{_mkey}{_cond_tag}.png"
                        save_bsdf_2d_png(_u, _v, _b, _bsdf_png, method=_mkey.upper())
                        _plot_paths.append(_bsdf_png)

                # インタラクティブ HTML（MLflow の artifact をクリック→新タブで動作）
                try:
                    from ..visualization.holoviews_plots import (
                        plot_bsdf_report,
                        plot_heightmap,
                        save_html,
                    )

                    # surface.html — 2D カラーマップ + ヒストグラム + 断面プロファイル
                    _surf_html = _td / "surface.html"
                    _surf_layout = plot_heightmap(
                        hm, title=_surf_title, unit="nm",
                    )
                    save_html(_surf_layout, _surf_html)
                    _plot_paths.append(_surf_html)

                    # bsdf_report.html — 多条件時は Tabs、実測行があれば自動オーバーレイ
                    _report_html = _td / "bsdf_report.html"
                    _report_layout = plot_bsdf_report(
                        df_combined,
                        metrics=all_metrics,
                        scale="log",
                        title=f"BSDF Report — {model_name}",
                    )
                    save_html(_report_layout, _report_html)
                    _plot_paths.append(_report_html)
                except Exception as _html_err:
                    logger.warning(
                        f"インタラクティブ HTML の生成に失敗（PNG のみ記録）: {_html_err}"
                    )

                run_id = ml_logger.log_trial(
                    params, all_metrics, df_combined, plot_paths=_plot_paths
                )
            logger.info(f"MLflow 記録完了: run_id={run_id}")

    logger.info("シミュレーション完了。")


# ── surface ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="設定ファイルパス（YAML）")
@click.option("--output", "-o", default=None, help="出力ファイルパス（省略時: surface.html または surface.png）")
@click.option("--format", "fmt", default="html", type=click.Choice(["html", "png"]), show_default=True, help="出力形式")
@click.option("--unit", default="nm", type=click.Choice(["nm", "um"]), show_default=True, help="高さの表示単位")
@click.option("--colormap", default="RdYlBu_r", show_default=True, help="カラーマップ名")
def surface(
    config: str,
    output: str | None,
    fmt: str,
    unit: str,
    colormap: str,
) -> None:
    """表面形状を 2D カラーマップ・ヒストグラム・断面プロファイルで可視化する。"""
    from ..io.config_loader import BSDFConfig
    from ..models import create_model_from_config, load_plugins
    from ..metrics.surface import compute_all_surface_metrics
    from ..visualization.holoviews_plots import plot_heightmap, save_heightmap_png, save_html

    cfg = BSDFConfig.from_file(config)
    load_plugins()

    out_path = output or f"surface.{fmt}"

    t0 = time.perf_counter()
    logger.info(f"[1/2] 高さマップ生成中... (grid={cfg.surface.get('grid_size', 4096)})")
    model = create_model_from_config(cfg._resolved)
    hm = model.get_height_map()
    logger.info(f"      完了 ({_elapsed(t0)})  {hm.grid_size}×{hm.grid_size}, pixel={hm.pixel_size_um}μm")

    # 表面形状指標を計算してタイトルに埋め込む
    surface_metrics = compute_all_surface_metrics(hm, verbose=True)
    sq_nm = surface_metrics["sq_um"] * 1000
    sa_nm = surface_metrics["sa_um"] * 1000
    model_name = cfg.surface.get("model", "")
    plot_title = f"{model_name}  Sq={sq_nm:.2f}nm  Sa={sa_nm:.2f}nm"

    logger.info(f"[2/2] 可視化出力中... ({fmt} → {out_path})")
    t0 = time.perf_counter()

    if fmt == "png":
        save_heightmap_png(
            hm, path=out_path,
            title=plot_title,
            colormap=colormap,
            unit=unit,
        )
    else:
        import panel as pn
        layout = plot_heightmap(hm, title=plot_title, colormap=colormap, unit=unit)
        save_html(layout, out_path)

    logger.info(f"      完了 ({_elapsed(t0)})  保存先: {out_path}")


# ── optimize ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="設定ファイルパス（YAML）")
@click.option("--trials", "-n", default=None, type=int, help="試行回数（設定ファイルの値を上書き）")
@click.option("--study-name", default=None, help="Optuna Study 名")
def optimize(config: str, trials: int | None, study_name: str | None) -> None:
    """Optuna による自動最適化を実行する。"""
    from ..io.config_loader import BSDFConfig
    from ..models import create_model_from_config, load_plugins
    from ..optics.fft_bsdf import compute_bsdf_fft
    from ..optics.psd_bsdf import compute_bsdf_psd
    from ..metrics.surface import compute_all_surface_metrics
    from ..metrics.optical import compute_all_optical_metrics
    from ..optimization.optuna_runner import BSDFOptimizer
    from ..optimization.mlflow_logger import RawDataLogger
    from ..io.parquet_schema import build_dataframe

    import optuna
    import pandas as pd
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = BSDFConfig.from_file(config)
    load_plugins()

    optuna_cfg = cfg.optuna
    n_trials = trials or optuna_cfg.get("n_trials", 100)
    n_jobs = optuna_cfg.get("n_jobs", 1)
    dup_cfg = optuna_cfg.get("duplicate_skip", {})
    dup_threshold = dup_cfg.get("distance_threshold", 0.01) if dup_cfg.get("enabled", True) else 999.0

    # 目的関数の設定（config.yaml の optuna.objectives から読み込む）
    default_objectives = [
        {"metric": "haze_fft_0_t",          "direction": "minimize"},
        {"metric": "sparkle_fft_555_0_t",   "direction": "minimize"},
    ]
    obj_cfg = optuna_cfg.get("objectives", default_objectives)
    directions = [o["direction"] for o in obj_cfg]

    # _ml 指標は optimize 未対応（Adding-Doubling は simulate のみ実装）
    ml_objectives = [
        o["metric"] for o in obj_cfg
        if o["metric"].endswith("_ml") or "_ml_" in o["metric"]
    ]
    if ml_objectives:
        logger.error(
            f"optimize の objectives に _ml 指標は指定できません: {ml_objectives}\n"
            "  Adding-Doubling（MultiLayer）は bsdf simulate のみ対応。\n"
            "  代わりに haze_fft_0_t / haze_psd_0_t などを使用してください。"
        )
        sys.exit(1)

    # 目的関数の指標名から必要な計算手法を自動判定（_fft_/_psd_/_ml_ をキーに検出）
    # 新命名規則 <name>_<method>_<deg>_<mode> では method が中央にあるため _{method}_ で判定
    needed_methods: set[str] = set()
    for o in obj_cfg:
        mname = o["metric"]
        if "_fft_" in mname or mname.endswith("_fft"):
            needed_methods.add("fft")
        elif "_psd_" in mname or mname.endswith("_psd"):
            needed_methods.add("psd")
    # 表面形状指標のみ指定された場合は FFT をデフォルトで実行
    if not needed_methods:
        needed_methods.add("fft")

    logger.info(f"最適化目的関数: {[o['metric'] for o in obj_cfg]}")
    logger.info(f"計算手法: {sorted(needed_methods)}")

    # 探索空間の定義（RandomRoughSurface の例）
    surface_model = cfg.surface.get("model", "RandomRoughSurface")
    if surface_model == "RandomRoughSurface":
        search_space = {
            "rq_um": (0.001, 0.1),
            "lc_um": (0.5, 20.0),
            "fractal_dim": (2.0, 3.0),
        }
    else:
        logger.error(f"optimize コマンドは現在 RandomRoughSurface のみ対応。モデル: {surface_model}")
        sys.exit(1)

    ml_logger = RawDataLogger(
        tracking_uri=cfg.mlflow.get("tracking_uri", "mlruns")
    )
    approx_mode = cfg._resolved.get("psd", {}).get("approx_mode", False)

    def objective(trial: "optuna.Trial") -> tuple[float, ...]:
        rq = trial.suggest_float("rq_um", *search_space["rq_um"])
        lc = trial.suggest_float("lc_um", *search_space["lc_um"])
        fd = trial.suggest_float("fractal_dim", *search_space["fractal_dim"])

        from ..models.random_rough import RandomRoughSurface
        model = RandomRoughSurface(
            rq_um=rq, lc_um=lc, fractal_dim=fd,
            grid_size=cfg.surface.get("grid_size", 4096),
            pixel_size_um=cfg.surface.get("pixel_size_um", 0.25),
        )
        hm = model.get_height_map()

        # 必要な手法のみ計算
        method_bsdf: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        all_dfs = []

        if "fft" in needed_methods:
            u_f, v_f, bsdf_f = compute_bsdf_fft(
                height_map=hm,
                wavelength_um=cfg.wavelength_um,
                theta_i_deg=cfg.theta_i_effective_deg,
                phi_i_deg=cfg.phi_i_deg,
                n1=cfg.n1,
                n2=cfg.n2,
                is_btdf=cfg.is_btdf,
                fft_mode=cfg.fft_mode,
                apply_fresnel=cfg.fft_apply_fresnel,
            )
            method_bsdf["fft"] = (u_f, v_f, bsdf_f)
            all_dfs.append(build_dataframe(
                u_f, v_f, bsdf_f, "FFT",
                cfg.theta_i_deg, cfg.phi_i_deg, cfg.wavelength_um, cfg.polarization,
                is_btdf=cfg.is_btdf,
            ))

        if "psd" in needed_methods:
            u_p, v_p, bsdf_p = compute_bsdf_psd(
                height_map=hm,
                wavelength_um=cfg.wavelength_um,
                theta_i_deg=cfg.theta_i_effective_deg,
                phi_i_deg=cfg.phi_i_deg,
                n1=cfg.n1,
                n2=cfg.n2,
                polarization=cfg.polarization,
                is_btdf=cfg.is_btdf,
                approx_mode=approx_mode,
            )
            method_bsdf["psd"] = (u_p, v_p, bsdf_p)
            all_dfs.append(build_dataframe(
                u_p, v_p, bsdf_p, "PSD",
                cfg.theta_i_deg, cfg.phi_i_deg, cfg.wavelength_um, cfg.polarization,
                is_btdf=cfg.is_btdf,
            ))

        # 表面形状指標（simulate と統一）
        surface_metrics = compute_all_surface_metrics(hm)

        # 光学指標（手法別サフィックス付き、simulate と統一）
        all_optical: dict[str, float] = {}
        for method_key, (u_m, v_m, bsdf_m) in method_bsdf.items():
            optical_m = compute_all_optical_metrics(
                u_grid=u_m, v_grid=v_m, bsdf=bsdf_m,
                config=cfg.metrics,
                bsdf_floor=cfg.bsdf_floor,
            )
            for k, val in optical_m.items():
                all_optical[f"{k}_{method_key}"] = val

        all_metrics = {**surface_metrics, **all_optical}

        # MLflow 記録
        df = pd.concat(all_dfs, ignore_index=True)
        ml_logger.log_trial(
            params={"rq_um": rq, "lc_um": lc, "fractal_dim": fd},
            metrics=all_metrics,
            df=df,
            run_name=f"trial_{trial.number}",
        )

        # 目的関数の値を config.yaml の objectives 順に返す
        return tuple(all_metrics.get(o["metric"], 0.0) for o in obj_cfg)

    optimizer = BSDFOptimizer(
        objective_fn=objective,
        search_space=search_space,
        directions=directions,
        n_trials=n_trials,
        n_jobs=n_jobs,
        duplicate_threshold=dup_threshold,
        study_name=study_name,
    )

    logger.info(f"最適化開始: {n_trials} 試行")
    study = optimizer.run()

    best = optimizer.best_trials_summary()
    logger.info(f"パレートフロント: {len(best)} 試行")
    for b in best:
        logger.info(f"  Trial #{b['trial_number']}: {b['params']} → {b['values']}")


# ── visualize ─────────────────────────────────────────────────────────────────

# ── runs（サブコマンドグループ）──────────────────────────────────────────────

@cli.group()
def runs() -> None:
    """MLflow run の一覧・検索（list サブコマンド）。"""


def _format_table(rows: list[list[str]], headers: list[str]) -> str:
    """プレーンテキストの整形テーブルを返す（rich 依存なし）。"""
    if not rows:
        return "（該当 run なし）"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append(sep.join("-" * w for w in widths))
    for row in rows:
        lines.append(sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


@runs.command("list")
@click.option("--tracking-uri", default="mlruns", show_default=True, help="MLflow トラッキング URI")
@click.option("--experiment", "-e", default=None,
              help="実験名（省略時: 01_BSDF_Raw_Data）")
@click.option("--sort-by", "-s", default=None,
              help="並び順の基準メトリクス名（例: haze_fft_0_t）。省略時は開始時刻降順")
@click.option("--ascending/--descending", default=True, show_default=True,
              help="sort-by 指定時の昇順 / 降順")
@click.option("--limit", "-n", default=20, show_default=True, type=int,
              help="最大表示件数")
@click.option("--metrics", "-m", default=None,
              help="カンマ区切りの表示メトリクス名（例: haze_fft_0_t,sparkle_fft_555_0_t）")
def runs_list(
    tracking_uri: str,
    experiment: str | None,
    sort_by: str | None,
    ascending: bool,
    limit: int,
    metrics: str | None,
) -> None:
    """MLflow の run 一覧を表示する（visualize の --run-id を選ぶため）。

    例:
      bsdf runs list --sort-by haze_fft_0_t --limit 10
      bsdf runs list -s sparkle_fft_555_0_t -m haze_fft_0_t,sparkle_fft_555_0_t --descending
      bsdf runs list -e 02_Analysis_Reports -n 5
    """
    from ..optimization.mlflow_logger import EXPERIMENT_RAW_DATA, list_runs

    exp_name = experiment or EXPERIMENT_RAW_DATA
    rows_raw = list_runs(
        tracking_uri=tracking_uri,
        experiment_name=exp_name,
        sort_by=sort_by,
        ascending=ascending,
        limit=limit,
    )
    if not rows_raw:
        click.echo(f"実験 '{exp_name}' に run が見つからない。")
        return

    # メトリクス列の決定
    if metrics:
        metric_cols = [m.strip() for m in metrics.split(",") if m.strip()]
    else:
        # sort_by を先頭に、使われそうな代表メトリクスを自動選択
        auto_preferred = [
            "haze_fft_0_t",
            "gloss_fft_20_r", "gloss_fft_60_r", "gloss_fft_85_r",
            "doi_nser_fft_0_t",
            "doi_comb_fft_0_t", "doi_comb_fft_0_r",
            "doi_astm_fft_20_r", "doi_astm_fft_30_r",
        ]
        metric_cols_set: list[str] = []
        if sort_by:
            metric_cols_set.append(sort_by)
        for m in auto_preferred:
            if m not in metric_cols_set:
                metric_cols_set.append(m)
        metric_cols = metric_cols_set

    import datetime as _dt

    headers = ["short_id", "started"] + metric_cols + ["run_name"]
    table_rows: list[list[str]] = []
    for r in rows_raw:
        short_id = r["run_id"][:8]
        started = _dt.datetime.fromtimestamp(r["start_time"] / 1000).strftime(
            "%Y-%m-%d %H:%M"
        )
        metric_vals = [
            (f"{r['metrics'][m]:.4g}" if m in r["metrics"] else "-")
            for m in metric_cols
        ]
        name = r["run_name"] or ""
        table_rows.append([short_id, started] + metric_vals + [name])

    click.echo(f"[実験: {exp_name}]  表示 {len(table_rows)} 件（最大 {limit}）")
    if sort_by:
        click.echo(f"sort_by: {sort_by} ({'昇順' if ascending else '降順'})")
    click.echo()
    click.echo(_format_table(table_rows, headers))
    click.echo()
    click.echo("使用例:")
    click.echo(
        "  bsdf visualize --run-id <short_id> --log-to-mlflow"
    )
    click.echo(
        "  bsdf visualize --run-id latest --log-to-mlflow"
    )
    if sort_by:
        click.echo(
            f"  bsdf visualize --run-id 'best:{sort_by}' --log-to-mlflow"
        )


# ── dashboard ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="設定ファイルパス（YAML）")
@click.option("--port", "-p", default=5006, show_default=True, type=int, help="Panel サーバーポート")
@click.option("--host", default="localhost", show_default=True,
              help="バインドアドレス。'0.0.0.0' で全インターフェース公開（他PCからアクセス可）")
@click.option("--preview-grid", default=512, show_default=True, type=int,
              help="プレビュー計算のグリッドサイズ（大きいほど精細・遅い）")
@click.option("--no-browser", is_flag=True, default=False,
              help="起動時にブラウザを自動で開かない")
def dashboard(config: str, port: int, host: str, preview_grid: int, no_browser: bool) -> None:
    """リアルタイム BSDF ダッシュボードをサーバーで起動する。

    config.yaml の `surface.model` に応じて以下のダッシュボードが起動する：
    - RandomRoughSurface: rq_um / lc_um / fractal_dim スライダー
    - SphericalArraySurface: radius / pitch / placement / overlap_mode
    - MeasuredSurface 系（DeviceVk6Surface 等）: 固定表示

    `measured_bsdf.path` が config に指定されていれば、条件一致する
    実測ブロックを 1D プロファイルに黒点 Scatter で自動オーバーレイする。

    他 PC からアクセスさせるには --host 0.0.0.0 を指定し、
    ファイアウォールでポート（デフォルト 5006）を開放すること。
    """
    from ..visualization.dynamicmap import create_dashboard_from_config

    logger.info(f"設定ファイル読み込み: {config}")
    dash = create_dashboard_from_config(
        config_path=config, preview_grid_size_idle=preview_grid,
    )
    display_host = host if host != "0.0.0.0" else "<このPCのIPアドレス>"
    logger.info(
        f"ダッシュボード起動: http://{display_host}:{port}  "
        f"(Ctrl+C で停止)"
    )
    dash.serve(port=port, show=(not no_browser), address=host)


# ── visualize ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--run-id", required=True,
              help="MLflow の run_id。'latest' / 'latest-N' / 'best:METRIC[:max]' / 8文字以上のプレフィックスも受理")
@click.option("--tracking-uri", default="mlruns", show_default=True, help="MLflow トラッキング URI")
@click.option("--experiment", "-e", default=None,
              help="run_id 解決時の実験名（省略時: 01_BSDF_Raw_Data）")
@click.option("--output", "-o", default="report.html", show_default=True, help="出力 HTML パス")
@click.option("--scale", default="log", type=click.Choice(["linear", "log"]), show_default=True)
@click.option("--log-to-mlflow/--no-log-to-mlflow", default=False, show_default=True,
              help="生成した HTML を元 run の artifacts/plots/ に書き戻す（optimize 後の探索用）")
def visualize(
    run_id: str,
    tracking_uri: str,
    experiment: str | None,
    output: str,
    scale: str,
    log_to_mlflow: bool,
) -> None:
    """MLflow の Run から BSDF レポート（1D/2D/指標テーブル）を HTML 出力する。

    --run-id は以下を受理:
      - 32 文字の完全な run_id
      - 'latest' / 'latest-2' / 'latest-5' — 最新から N 番目
      - 'best:haze_fft_0_t' — metric の最小値 run
      - 'best:sparkle_fft_555_0_t:max' — metric の最大値 run
      - 8 文字以上のプレフィックス（例: 'a1b2c3d4'）

    `--log-to-mlflow` を指定すると、生成した HTML を元 run の
    `artifacts/plots/` に書き戻す。optimize で多数の run が出たあと、
    気になる run にだけレポートを追記してブラウザでクリック閲覧する用途。
    既に同名の artifact があれば上書きされる（MLflow の挙動に準拠）。
    """
    from ..optimization.mlflow_logger import (
        EXPERIMENT_RAW_DATA,
        load_trial_dataframe,
        load_trial_metrics,
        resolve_run_id,
    )
    from ..visualization.holoviews_plots import plot_bsdf_report, save_html

    # ショートカット / プレフィックスを完全な run_id に解決
    exp_name = experiment or EXPERIMENT_RAW_DATA
    try:
        resolved_run_id = resolve_run_id(
            run_id, tracking_uri=tracking_uri, experiment_name=exp_name,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    if resolved_run_id != run_id:
        logger.info(f"run_id 解決: '{run_id}' → {resolved_run_id}")
    run_id = resolved_run_id

    logger.info(f"run_id={run_id} のデータを読み込み中...")
    df = load_trial_dataframe(run_id, tracking_uri=tracking_uri)

    logger.info("MLflow metrics を取得中...")
    try:
        metrics = load_trial_metrics(run_id, tracking_uri=tracking_uri)
    except Exception:
        metrics = None
        logger.warning("metrics の取得に失敗しました。テーブルは省略されます。")

    plot = plot_bsdf_report(
        df,
        metrics=metrics,
        scale=scale,
        title=f"BSDF Report — run_id: {run_id[:8]}",
    )
    save_html(plot, output)
    logger.info(f"HTML 保存: {output}")

    if log_to_mlflow:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        try:
            client.log_artifact(run_id, local_path=str(output), artifact_path="plots")
            html_name = Path(output).name
            logger.info(
                f"MLflow 書き戻し完了: run_id={run_id} → artifacts/plots/{html_name}"
            )
            logger.info(
                "  MLflow UI の当該 run の artifacts タブからクリックで閲覧可能。"
            )
        except Exception as e:
            logger.error(f"MLflow への書き戻しに失敗しました: {e}")
            raise click.ClickException(f"MLflow artifact 書き戻し失敗: {e}")


# ── report ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--run-ids", required=True, help="カンマ区切りの run_id リスト")
@click.option("--tracking-uri", default="mlruns", show_default=True, help="MLflow トラッキング URI")
@click.option("--output", "-o", default="comparison_report.html", show_default=True, help="出力 HTML パス")
@click.option("--log-to-mlflow/--no-log-to-mlflow", default=True, show_default=True)
def report(run_ids: str, tracking_uri: str, output: str, log_to_mlflow: bool) -> None:
    """複数 Run の BSDF を重ね合わせた比較レポートを生成する。"""
    import pandas as pd
    from ..optimization.mlflow_logger import load_trial_dataframe, AnalysisLogger
    from ..visualization.holoviews_plots import plot_bsdf_1d_overlay, save_html

    run_id_list = [r.strip() for r in run_ids.split(",")]
    logger.info(f"{len(run_id_list)} 件の Run を読み込み中...")

    dfs = []
    for rid in run_id_list:
        df = load_trial_dataframe(rid, tracking_uri=tracking_uri)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    plot = plot_bsdf_1d_overlay(combined, scale="log", title="BSDF 多Run比較")
    save_html(plot, output)

    if log_to_mlflow:
        from ..optimization.mlflow_logger import AnalysisLogger
        analysis_logger = AnalysisLogger(tracking_uri=tracking_uri)
        report_run_id = analysis_logger.log_report(
            run_ids=run_id_list,
            html_path=output,
            report_name="comparison_report",
        )
        logger.info(f"MLflow 記録完了: run_id={report_run_id}")

    logger.info(f"比較レポート保存: {output}")


if __name__ == "__main__":
    cli()
