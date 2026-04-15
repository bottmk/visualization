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
        plot_paths: "list[str | Path] | None" = None,
    ) -> str:
        """1 Trial の結果を MLflow に記録する。

        Args:
            params: 形状パラメータ辞書
            metrics: 評価指標辞書（haze_fft, gloss_psd 等）
            df: BSDF データ DataFrame（Parquet として保存）
            run_name: Run の名前（省略時は自動生成）
            plot_paths: artifacts/plots/ に保存する画像パスのリスト（省略可）

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

            # プロット画像の保存（オプション）
            if plot_paths:
                for p in plot_paths:
                    mlflow.log_artifact(str(p), artifact_path="plots")

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


def load_trial_metrics(run_id: str, tracking_uri: str = "mlruns") -> dict[str, float]:
    """01_BSDF_Raw_Data の Run から metrics 辞書を取得する。

    Args:
        run_id: MLflow の run_id
        tracking_uri: MLflow トラッキング URI

    Returns:
        metrics 辞書（キー: 指標名、値: float）
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return dict(run.data.metrics)


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


# ── run_id 解決ヘルパー（ショートカット名・プレフィックス対応）─────────────────

def list_runs(
    tracking_uri: str = "mlruns",
    experiment_name: str = EXPERIMENT_RAW_DATA,
    sort_by: str | None = None,
    ascending: bool = True,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """実験内の run 一覧を取得する。

    Args:
        tracking_uri: MLflow トラッキング URI
        experiment_name: 実験名（デフォルト: 01_BSDF_Raw_Data）
        sort_by: 並び順の基準メトリクス名（例: 'haze_fft'）。None → 開始時刻降順
        ascending: sort_by 指定時に昇順にするか（最小値が先）
        limit: 最大件数

    Returns:
        run 辞書のリスト。各要素のキー:
          run_id, start_time, metrics (dict), params (dict), run_name
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []

    if sort_by:
        order = f"metrics.{sort_by} {'ASC' if ascending else 'DESC'}"
    else:
        order = "attributes.start_time DESC"

    runs = client.search_runs(
        exp.experiment_id,
        max_results=limit,
        order_by=[order],
    )
    return [
        {
            "run_id": r.info.run_id,
            "start_time": r.info.start_time,
            "metrics": dict(r.data.metrics),
            "params": dict(r.data.params),
            "run_name": r.data.tags.get("mlflow.runName", ""),
        }
        for r in runs
    ]


def resolve_run_id(
    ref: str,
    tracking_uri: str = "mlruns",
    experiment_name: str = EXPERIMENT_RAW_DATA,
) -> str:
    """ショートカット名・プレフィックスを実 run_id に解決する。

    サポートする形式:
      - 完全な run_id（32 文字）: そのまま返す
      - 'latest' / 'latest-N': 最新から N 個目（0-index ではなく 1-index、'latest' = 'latest-1'）
      - 'best:METRIC' / 'best:METRIC:min': 指定 metric が最小の run
      - 'best:METRIC:max': 指定 metric が最大の run
      - 8 文字以上のプレフィックス: ユニークに特定できれば採用、複数マッチはエラー

    Args:
        ref: run_id または解決対象のショートカット文字列
        tracking_uri: MLflow トラッキング URI
        experiment_name: 検索対象の実験名

    Returns:
        32 文字の完全な run_id

    Raises:
        ValueError: 解決に失敗した場合（該当 run なし、複数マッチ、未知の形式等）
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # 完全一致（32 文字の 16 進）: そのまま返す
    if len(ref) == 32 and all(c in "0123456789abcdefABCDEF" for c in ref):
        return ref

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(
            f"実験 '{experiment_name}' が見つからない。tracking_uri={tracking_uri}"
        )

    # latest / latest-N
    if ref == "latest" or ref.startswith("latest-"):
        n = 1
        if ref != "latest":
            try:
                n = int(ref.split("-", 1)[1])
            except ValueError as e:
                raise ValueError(f"latest-N の N が整数でない: '{ref}'") from e
        if n < 1:
            raise ValueError(f"latest-N の N は 1 以上: '{ref}'")
        runs = client.search_runs(
            exp.experiment_id,
            max_results=n,
            order_by=["attributes.start_time DESC"],
        )
        if len(runs) < n:
            raise ValueError(
                f"実験 '{experiment_name}' の run が {len(runs)} 件しかない（要求: {n} 番目）"
            )
        return runs[n - 1].info.run_id

    # best:METRIC[:min|:max]
    if ref.startswith("best:"):
        parts = ref.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"'best:METRIC' または 'best:METRIC:max' の形式で指定: '{ref}'"
            )
        metric_name = parts[1]
        direction = parts[2] if len(parts) == 3 else "min"
        if direction not in ("min", "max"):
            raise ValueError(
                f"best の方向は 'min' または 'max': '{ref}'"
            )
        order = (
            f"metrics.{metric_name} {'ASC' if direction == 'min' else 'DESC'}"
        )
        # metric がある run だけをフィルタ（MLflow の search_runs は metric
        # を持たない run も返すため、手動で絞り込む必要がある）
        runs = client.search_runs(
            exp.experiment_id, max_results=10000, order_by=[order],
        )
        runs_with_metric = [r for r in runs if metric_name in r.data.metrics]
        if not runs_with_metric:
            raise ValueError(
                f"metric '{metric_name}' を持つ run が実験 "
                f"'{experiment_name}' に見つからない"
            )
        return runs_with_metric[0].info.run_id

    # プレフィックスマッチ（8 文字以上）
    if len(ref) >= 8:
        # 全 run を取得してプレフィックスでフィルタ
        runs = client.search_runs(
            exp.experiment_id, max_results=10000,
        )
        matches = [r for r in runs if r.info.run_id.startswith(ref)]
        if len(matches) == 1:
            return matches[0].info.run_id
        if len(matches) == 0:
            raise ValueError(
                f"プレフィックス '{ref}' に一致する run が見つからない"
            )
        raise ValueError(
            f"プレフィックス '{ref}' に複数の run がマッチした "
            f"（{len(matches)} 件）。もっと長い文字列を指定してください。"
        )

    raise ValueError(
        f"run_id の解決に失敗: '{ref}'。32 文字の run_id、"
        "'latest' / 'latest-N'、'best:METRIC' / 'best:METRIC:max'、"
        "または 8 文字以上のプレフィックスを指定してください。"
    )
