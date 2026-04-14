"""Optuna による自動最適化。

spec_main.md Section 6.1:
- 多目的最適化（MOTPE）
- タグチメソッドとの融合（add_trial による知識注入）
- 正規化ユークリッド距離による重複試行スキップ
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import optuna

logger = logging.getLogger(__name__)


# ── 重複試行スキップ ──────────────────────────────────────────────────────────

def _normalize_params(
    params: dict[str, float],
    search_space: dict[str, tuple[float, float]],
) -> np.ndarray:
    """パラメータを [0, 1] に正規化した配列を返す。

    Args:
        params: パラメータ辞書
        search_space: {パラメータ名: (low, high)} の辞書

    Returns:
        正規化済みパラメータ配列
    """
    vec = []
    for name, (low, high) in search_space.items():
        val = params.get(name, (low + high) / 2)
        normalized = (val - low) / (high - low) if high > low else 0.0
        vec.append(float(np.clip(normalized, 0.0, 1.0)))
    return np.array(vec)


def is_duplicate(
    candidate: dict[str, float],
    existing_trials: list[optuna.trial.FrozenTrial],
    search_space: dict[str, tuple[float, float]],
    threshold: float = 0.01,
) -> bool:
    """候補パラメータが過去の試行と重複しているかどうかを判定する。

    正規化ユークリッド距離で評価し、閾値以内の場合は重複とみなす。

    Args:
        candidate: 候補パラメータ辞書
        existing_trials: 過去の試行リスト
        search_space: {パラメータ名: (low, high)} の辞書
        threshold: 距離閾値（デフォルト: 0.01）

    Returns:
        True の場合は重複（スキップすべき）
    """
    if not existing_trials:
        return False

    cand_vec = _normalize_params(candidate, search_space)

    for trial in existing_trials:
        if trial.state not in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        ):
            continue
        trial_params = {
            k: v for k, v in trial.params.items() if k in search_space
        }
        if len(trial_params) < len(search_space):
            continue
        trial_vec = _normalize_params(trial_params, search_space)
        dist = float(np.linalg.norm(cand_vec - trial_vec))
        if dist < threshold:
            return True

    return False


# ── Optuna ランナー ───────────────────────────────────────────────────────────

class BSDFOptimizer:
    """BSDFシミュレーションの Optuna 最適化ランナー。

    Args:
        objective_fn: 目的関数。Trial を受け取り (metric1, metric2, ...) のタプルを返す。
        search_space: {パラメータ名: (low, high)} の探索空間定義。
        directions: 最適化方向のリスト（'minimize' / 'maximize'）。
        n_trials: 試行回数。
        n_jobs: 並列ジョブ数。
        sampler: サンプラー名（'MOTPE' / 'TPE'）。
        duplicate_threshold: 重複スキップの距離閾値。
        study_name: Optuna Study 名（None の場合は自動生成）。
    """

    def __init__(
        self,
        objective_fn: Callable[[optuna.Trial], tuple[float, ...]],
        search_space: dict[str, tuple[float, float]],
        directions: list[str] | None = None,
        n_trials: int = 200,
        n_jobs: int = 1,
        sampler: str = "MOTPE",
        duplicate_threshold: float = 0.01,
        study_name: str | None = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.search_space = search_space
        self.directions = directions or ["minimize"]
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.duplicate_threshold = duplicate_threshold
        self.study_name = study_name

        # サンプラーの選択
        if sampler == "MOTPE" and len(self.directions) > 1:
            self._sampler = optuna.samplers.TPESampler(multivariate=True)
        else:
            self._sampler = optuna.samplers.TPESampler()

        # Study の作成
        self.study = optuna.create_study(
            directions=self.directions,
            sampler=self._sampler,
            study_name=self.study_name,
        )

    def _wrapped_objective(self, trial: optuna.Trial) -> tuple[float, ...]:
        """重複スキップを組み込んだラッパー目的関数。"""
        # 候補パラメータをサジェスト
        params = {
            name: trial.suggest_float(name, low, high)
            for name, (low, high) in self.search_space.items()
        }

        # 重複チェック
        if is_duplicate(
            params,
            self.study.trials,
            self.search_space,
            self.duplicate_threshold,
        ):
            logger.info(f"Trial #{trial.number}: 重複パラメータのためスキップ。")
            raise optuna.exceptions.TrialPruned()

        return self.objective_fn(trial)

    def add_taguchi_trials(self, trials_data: list[dict[str, Any]]) -> None:
        """タグチメソッドの実験データを Optuna のナレッジベースとして注入する。

        Args:
            trials_data: [{'params': {...}, 'values': [v1, v2, ...]}] のリスト
        """
        for data in trials_data:
            distributions = {
                name: optuna.distributions.FloatDistribution(low, high)
                for name, (low, high) in self.search_space.items()
            }
            trial = optuna.trial.create_trial(
                params=data["params"],
                distributions=distributions,
                values=data["values"],
            )
            self.study.add_trial(trial)
            logger.info(f"タグチ試行を注入: params={data['params']}")

    def enqueue_trial(self, params: dict[str, float]) -> None:
        """優先実行する試行をキューに追加する。

        Args:
            params: 実行したいパラメータ辞書
        """
        self.study.enqueue_trial(params)

    def run(self) -> optuna.Study:
        """最適化を実行する。

        Returns:
            完了した Optuna Study
        """
        logger.info(
            f"最適化開始: n_trials={self.n_trials}, n_jobs={self.n_jobs}, "
            f"directions={self.directions}"
        )
        self.study.optimize(
            self._wrapped_objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            catch=(Exception,),
        )
        logger.info(f"最適化完了: {len(self.study.trials)} 試行")
        return self.study

    def best_trials_summary(self) -> list[dict[str, Any]]:
        """パレートフロントの最良試行のサマリを返す。

        Returns:
            [{'trial_number': ..., 'params': ..., 'values': ...}] のリスト
        """
        if len(self.directions) > 1:
            best = self.study.best_trials
        else:
            best = [self.study.best_trial]

        return [
            {
                "trial_number": t.number,
                "params": t.params,
                "values": t.values,
            }
            for t in best
        ]
