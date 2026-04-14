"""MLflow による実験管理（3階層 Experiment 管理）。

spec_main.md Section 6.2:
01_BSDF_Raw_Data: 1 Run = 1 形状
02_Analysis_Reports: 1 Run = 1 解析タスク
03_GenAI_Insights: LLM 考察レポート
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


EXPERIMENT_RAW_DATA    = "01_BSDF_Raw_Data"
EXPERIMENT_ANALYSIS    = "02_Analysis_Reports"
EXPERIMENT_GENAI       = "03_GenAI_Insights"


def _get_or_create_experiment(name: str) -> str:
    """Experiment を取得または作成して experiment_id を返す。"""
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id


class RawDataLogger:
    """01_BSDF_Raw_Data への記録（1 Run = 1 形状）。"""

    def __init__(self, tracking_uri: str = "mlruns") -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._exp_id = _get_or_create_experiment(EXPERIMENT_RAW_DATA)

    def log_trial(
        self,
        params: dict[str, Any],
        metrics: dict[str, float],
        df: pd.DataFrame,
        run_name: str | None = None,
        heatmap_path: str | Path | None = None,
    ) -> str:
        """1 Trial の結果を MLflow に記録する。

        Args:
            params: 形状パラメータ辞書
            metrics: 評価指標辞書（haze, gloss, log_rmse 等）
            df: BSDF データ DataFrame（Parquet として保存）
            run_name: Run の名前（省略時は自動生成）
            heatmap_path: 2D ヒートマップ画像のパス（省略可）

        Returns:
            MLflow の run_id
        """
        with mlflow.start_run(
            experiment_id=self._exp_id,
            run_name=run_name,
        ) as run:
            # パラメータの記録
            mlflow.log_params(params)

            # 評価指標の記録
            mlflow.log_metrics(metrics)

            # Parquet ファイルの保存
            with tempfile.TemporaryDirectory() as tmpdir:
                parquet_path = Path(tmpdir) / "bsdf_data.parquet"
                df.to_parquet(parquet_path, index=False, engine="pyarrow")
                mlflow.log_artifact(str(parquet_path), artifact_path="data")

            # ヒートマップ画像の保存（オプション）
            if heatmap_path is not None:
                mlflow.log_artifact(str(heatmap_path), artifact_path="plots")

            return run.info.run_id


class AnalysisLogger:
    """02_Analysis_Reports への記録（1 Run = 1 解析タスク）。"""

    def __init__(self, tracking_uri: str = "mlruns") -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._exp_id = _get_or_create_experiment(EXPERIMENT_ANALYSIS)

    def log_report(
        self,
        run_ids: list[str],
        html_path: str | Path,
        report_name: str | None = None,
        metrics_summary: dict[str, float] | None = None,
    ) -> str:
        """比較レポートを MLflow に記録する。

        Args:
            run_ids: 比較対象の 01_BSDF_Raw_Data の run_id リスト
            html_path: インタラクティブ HTML グラフのパス
            report_name: Run の名前
            metrics_summary: サマリ指標辞書

        Returns:
            MLflow の run_id
        """
        with mlflow.start_run(
            experiment_id=self._exp_id,
            run_name=report_name,
        ) as run:
            mlflow.log_param("source_run_ids", ",".join(run_ids))

            if metrics_summary:
                mlflow.log_metrics(metrics_summary)

            mlflow.log_artifact(str(html_path), artifact_path="reports")

            return run.info.run_id


def load_trial_dataframe(run_id: str, tracking_uri: str = "mlruns") -> pd.DataFrame:
    """01_BSDF_Raw_Data の Run から BSDF DataFrame を読み込む。

    Args:
        run_id: MLflow の run_id
        tracking_uri: MLflow トラッキング URI

    Returns:
        BSDF データ DataFrame
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, path="data")

    with tempfile.TemporaryDirectory() as tmpdir:
        for artifact in artifacts:
            if artifact.path.endswith(".parquet"):
                local_path = client.download_artifacts(run_id, artifact.path, tmpdir)
                return pd.read_parquet(local_path, engine="pyarrow")

    raise FileNotFoundError(f"run_id={run_id} に Parquet ファイルが見つからない。")
