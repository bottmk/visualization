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
def simulate(
    config: str,
    output_dir: str,
    method: str,
    save_parquet: bool,
    log_to_mlflow: bool,
) -> None:
    """BSDF シミュレーションを単体実行する。"""
    from ..io.config_loader import BSDFConfig
    from ..models import create_model_from_config, load_plugins
    from ..optics.fft_bsdf import compute_bsdf_fft
    from ..optics.psd_bsdf import compute_bsdf_psd
    from ..io.parquet_schema import build_dataframe, save_parquet as _save_parquet
    from ..metrics.surface import compute_all_surface_metrics
    from ..metrics.optical import compute_all_optical_metrics

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

    all_dfs = []
    approx_mode = cfg._resolved.get("psd", {}).get("approx_mode", False)

    # 光学指標計算に使う u, v, bsdf を明示的に追跡（method によらず正しい変数を使う）
    u_primary: np.ndarray | None = None
    v_primary: np.ndarray | None = None
    bsdf_primary: np.ndarray | None = None

    # [3] BSDF 計算
    step3_label = {"fft": "FFT", "psd": "PSD", "both": "FFT + PSD"}[method]
    logger.info(f"[3/4] {step3_label} 計算中...")

    # FFT 計算
    if method in ("fft", "both"):
        t0 = time.perf_counter()
        logger.info("      FFT 法...")
        u, v, bsdf_fft = compute_bsdf_fft(
            height_map=hm,
            wavelength_um=cfg.wavelength_um,
            theta_i_deg=cfg.theta_i_effective_deg,
            phi_i_deg=cfg.phi_i_deg,
            n1=cfg.n1,
            n2=cfg.n2,
            polarization=cfg.polarization,
            is_btdf=cfg.is_btdf,
        )
        u_primary, v_primary, bsdf_primary = u, v, bsdf_fft
        df_fft = build_dataframe(
            u, v, bsdf_fft, "FFT",
            cfg.theta_i_deg, cfg.phi_i_deg, cfg.wavelength_um, cfg.polarization,
            is_btdf=cfg.is_btdf,
        )
        all_dfs.append(df_fft)
        logger.info(f"      FFT 完了 ({_elapsed(t0)})")

    # PSD 計算
    if method in ("psd", "both"):
        t0 = time.perf_counter()
        logger.info(f"      PSD 法 (approx_mode={approx_mode})...")
        u, v, bsdf_psd = compute_bsdf_psd(
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
        if u_primary is None:  # FFT なし（method='psd'）の場合は PSD を primary に
            u_primary, v_primary, bsdf_primary = u, v, bsdf_psd
        df_psd = build_dataframe(
            u, v, bsdf_psd, "PSD",
            cfg.theta_i_deg, cfg.phi_i_deg, cfg.wavelength_um, cfg.polarization,
            is_btdf=cfg.is_btdf,
        )
        all_dfs.append(df_psd)
        logger.info(f"      PSD 完了 ({_elapsed(t0)})")

    # Adding-Doubling 多層合成（config で enabled: true の場合のみ）
    if cfg.adding_doubling.get("enabled", False) and bsdf_primary is not None:
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
        df_ml = build_dataframe(
            u_ml, v_ml, bsdf_ml, "MultiLayer",
            cfg.theta_i_deg, cfg.phi_i_deg, cfg.wavelength_um, cfg.polarization,
            is_btdf=cfg.is_btdf,
        )
        all_dfs.append(df_ml)
        u_primary, v_primary, bsdf_primary = u_ml, v_ml, bsdf_ml
        logger.info("Adding-Doubling 完了。")

    if all_dfs and bsdf_primary is not None:
        import pandas as pd
        df_combined = pd.concat(all_dfs, ignore_index=True)

        # [4] 光学指標
        enabled_metrics = [
            k for k in ("haze", "gloss", "doi", "sparkle")
            if cfg.metrics.get(k) is not None and cfg.metrics[k].get("enabled", True)
        ]
        logger.info(f"[4/4] 光学指標計算中... ({', '.join(enabled_metrics) or 'なし'})")
        t0 = time.perf_counter()
        optical_metrics = compute_all_optical_metrics(
            u_grid=u_primary,
            v_grid=v_primary,
            bsdf=bsdf_primary,
            config=cfg.metrics,
            bsdf_floor=cfg.bsdf_floor,
        )
        logger.info(f"      完了 ({_elapsed(t0)})")
        for k, val in optical_metrics.items():
            logger.info(f"        {k} = {val:.6f}")

        # Parquet 保存
        if save_parquet:
            parquet_path = out_path / "bsdf_data.parquet"
            _save_parquet(df_combined, parquet_path)
            logger.info(f"Parquet 保存: {parquet_path}")

        # MLflow 記録
        if log_to_mlflow:
            from ..optimization.mlflow_logger import RawDataLogger
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
            all_metrics = {**surface_metrics, **optical_metrics}
            run_id = ml_logger.log_trial(params, all_metrics, df_combined)
            logger.info(f"MLflow 記録完了: run_id={run_id}")

    logger.info("シミュレーション完了。")


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
    from ..metrics.optical import compute_log_rmse, compute_haze, compute_sparkle
    from ..optimization.optuna_runner import BSDFOptimizer
    from ..optimization.mlflow_logger import RawDataLogger
    from ..io.parquet_schema import build_dataframe

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = BSDFConfig.from_file(config)
    load_plugins()

    optuna_cfg = cfg.optuna
    n_trials = trials or optuna_cfg.get("n_trials", 100)
    n_jobs = optuna_cfg.get("n_jobs", 1)
    dup_cfg = optuna_cfg.get("duplicate_skip", {})
    dup_threshold = dup_cfg.get("distance_threshold", 0.01) if dup_cfg.get("enabled", True) else 999.0

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

    def objective(trial: "optuna.Trial") -> tuple[float, float]:
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

        u, v, bsdf = compute_bsdf_fft(
            height_map=hm,
            wavelength_um=cfg.wavelength_um,
            theta_i_deg=cfg.theta_i_effective_deg,
            phi_i_deg=cfg.phi_i_deg,
            n1=cfg.n1,
            n2=cfg.n2,
            is_btdf=cfg.is_btdf,
        )

        haze_val = compute_haze(u, v, bsdf)
        sparkle_val = compute_sparkle(u, v, bsdf, cfg.metrics.get("sparkle", {}))

        # MLflow 記録
        df = build_dataframe(
            u, v, bsdf, "FFT",
            cfg.theta_i_deg, cfg.phi_i_deg, cfg.wavelength_um, cfg.polarization,
            is_btdf=cfg.is_btdf,
        )
        ml_logger.log_trial(
            params={"rq_um": rq, "lc_um": lc, "fractal_dim": fd},
            metrics={"haze": haze_val, "sparkle": sparkle_val},
            df=df,
            run_name=f"trial_{trial.number}",
        )

        return haze_val, sparkle_val

    optimizer = BSDFOptimizer(
        objective_fn=objective,
        search_space=search_space,
        directions=["minimize", "minimize"],
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

@cli.command()
@click.option("--run-id", required=True, help="MLflow の run_id")
@click.option("--tracking-uri", default="mlruns", show_default=True, help="MLflow トラッキング URI")
@click.option("--output", "-o", default="report.html", show_default=True, help="出力 HTML パス")
@click.option("--scale", default="log", type=click.Choice(["linear", "log"]), show_default=True)
def visualize(run_id: str, tracking_uri: str, output: str, scale: str) -> None:
    """MLflow の Run から BSDF をプロットし HTML を出力する。"""
    from ..optimization.mlflow_logger import load_trial_dataframe
    from ..visualization.holoviews_plots import plot_bsdf_1d_overlay, save_html

    logger.info(f"run_id={run_id} のデータを読み込み中...")
    df = load_trial_dataframe(run_id, tracking_uri=tracking_uri)

    plot = plot_bsdf_1d_overlay(df, scale=scale, title=f"BSDF - run_id: {run_id[:8]}")
    save_html(plot, output)
    logger.info(f"HTML 保存: {output}")


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
