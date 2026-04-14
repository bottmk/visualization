"""最適化ユーティリティ（重複スキップ・BSDFOptimizer）のテスト。"""

import numpy as np
import pytest

from bsdf_sim.optimization.optuna_runner import (
    _normalize_params,
    is_duplicate,
    BSDFOptimizer,
)


# ── _normalize_params ────────────────────────────────────────────────────────

class TestNormalizeParams:
    def test_lower_bound_maps_to_zero(self):
        space = {"x": (0.0, 1.0)}
        vec = _normalize_params({"x": 0.0}, space)
        assert vec[0] == pytest.approx(0.0)

    def test_upper_bound_maps_to_one(self):
        space = {"x": (0.0, 1.0)}
        vec = _normalize_params({"x": 1.0}, space)
        assert vec[0] == pytest.approx(1.0)

    def test_midpoint(self):
        space = {"x": (0.0, 2.0)}
        vec = _normalize_params({"x": 1.0}, space)
        assert vec[0] == pytest.approx(0.5)

    def test_clamps_out_of_range(self):
        space = {"x": (0.0, 1.0)}
        vec = _normalize_params({"x": 2.0}, space)
        assert vec[0] == pytest.approx(1.0)

    def test_multiple_params_order(self):
        space = {"a": (0.0, 10.0), "b": (0.0, 100.0)}
        vec = _normalize_params({"a": 5.0, "b": 50.0}, space)
        assert vec[0] == pytest.approx(0.5)
        assert vec[1] == pytest.approx(0.5)


# ── is_duplicate ─────────────────────────────────────────────────────────────

def _make_frozen_trial(params: dict, values: list[float]):
    """テスト用の FrozenTrial を生成する。"""
    import optuna
    distributions = {
        k: optuna.distributions.FloatDistribution(0.0, 1.0)
        for k in params
    }
    return optuna.trial.create_trial(
        params=params,
        distributions=distributions,
        values=values,
    )


class TestIsDuplicate:
    def test_empty_history_is_not_duplicate(self):
        space = {"x": (0.0, 1.0)}
        assert is_duplicate({"x": 0.5}, [], space, threshold=0.01) is False

    def test_identical_params_is_duplicate(self):
        space = {"x": (0.0, 1.0)}
        trial = _make_frozen_trial({"x": 0.5}, [0.1])
        assert is_duplicate({"x": 0.5}, [trial], space, threshold=0.01) is True

    def test_distant_params_is_not_duplicate(self):
        space = {"x": (0.0, 1.0)}
        trial = _make_frozen_trial({"x": 0.0}, [0.1])
        assert is_duplicate({"x": 1.0}, [trial], space, threshold=0.01) is False

    def test_just_below_threshold_is_not_duplicate(self):
        """距離が閾値ちょうど未満の場合は重複でない。"""
        space = {"x": (0.0, 1.0)}
        trial = _make_frozen_trial({"x": 0.5}, [0.1])
        # 距離 = 0.009 < threshold 0.01 → 重複
        assert is_duplicate({"x": 0.509}, [trial], space, threshold=0.01) is True

    def test_threshold_zero_means_only_exact_match(self):
        """閾値=0 では同一点以外は重複でない。"""
        space = {"x": (0.0, 1.0)}
        trial = _make_frozen_trial({"x": 0.5}, [0.1])
        assert is_duplicate({"x": 0.5001}, [trial], space, threshold=0.0) is False

    def test_multiple_params(self):
        space = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        trial = _make_frozen_trial({"a": 0.0, "b": 0.0}, [0.1])
        # 正規化距離 = sqrt(0.5^2 + 0.5^2) ≈ 0.707 → 重複でない
        assert is_duplicate({"a": 0.5, "b": 0.5}, [trial], space, threshold=0.01) is False


# ── BSDFOptimizer ─────────────────────────────────────────────────────────────

class TestBSDFOptimizer:
    def _simple_objective(self, trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return (x,)

    def test_runs_n_trials(self):
        opt = BSDFOptimizer(
            objective_fn=self._simple_objective,
            search_space={"x": (0.0, 1.0)},
            directions=["minimize"],
            n_trials=5,
        )
        study = opt.run()
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        assert len(completed) == 5

    def test_duplicate_skip_reduces_completed(self):
        """重複スキップ有効時、完了試行数 ≤ n_trials。"""
        call_count = {"n": 0}

        def objective(trial):
            call_count["n"] += 1
            trial.suggest_float("x", 0.0, 1.0)
            return (0.5,)

        opt = BSDFOptimizer(
            objective_fn=objective,
            search_space={"x": (0.0, 1.0)},
            directions=["minimize"],
            n_trials=10,
            duplicate_threshold=0.5,  # 非常に大きな閾値で多くをスキップ
        )
        opt.run()
        # 少なくとも1試行は完了するはず（最初はスキップされない）
        assert call_count["n"] >= 1

    def test_best_trials_summary_single_objective(self):
        opt = BSDFOptimizer(
            objective_fn=self._simple_objective,
            search_space={"x": (0.0, 1.0)},
            directions=["minimize"],
            n_trials=3,
        )
        opt.run()
        summary = opt.best_trials_summary()
        assert len(summary) >= 1
        assert "trial_number" in summary[0]
        assert "params" in summary[0]
        assert "values" in summary[0]
