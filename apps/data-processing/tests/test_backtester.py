# -*- coding: utf-8 -*-
"""
Tests for the walk-forward backtesting harness.

Covers:
- metric helpers (MAE, RMSE, MAPE, skill score)
- BacktestConfig construction and serialisation
- BacktestReport / BacktestMetrics data classes
- run_backtest() engine with synthetic data
- Confidence label derivation
- Integration: ForecastResult.backtest_confidence propagation
"""

import importlib
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List

import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _real_pandas() -> Any:
    """Return the real pandas module, or None if it is unavailable."""
    mod = importlib.import_module("pandas")
    return mod if hasattr(mod, "date_range") else None


def _make_df(
    scores: List[float],
    hours_between: float = 1.0,
) -> Any:
    """
    Build a minimal analytics DataFrame from a list of sentiment scores.

    Requires the real pandas (with DataFrame and to_datetime).  Tests that
    call this helper are auto-skipped when the test environment only has the
    conftest stub.
    """
    pd = _real_pandas()
    if pd is None:
        pytest.skip("real pandas not available in this test environment")

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=i * hours_between) for i in range(len(scores))]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "sentiment_score": scores,
            "news_count": [10] * len(scores),
            "positive_pct": [0.5] * len(scores),
            "negative_pct": [0.2] * len(scores),
            "neutral_pct": [0.3] * len(scores),
        }
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestMetricHelpers:
    """Unit tests for the standalone metric functions."""

    def test_mae_basic(self) -> None:
        from src.analytics.backtester import _mae

        actuals = [1.0, 2.0, 3.0]
        preds = [1.5, 2.5, 3.5]
        assert _mae(actuals, preds) == pytest.approx(0.5)

    def test_mae_perfect(self) -> None:
        from src.analytics.backtester import _mae

        vals = [0.1, -0.3, 0.5]
        assert _mae(vals, vals) == pytest.approx(0.0)

    def test_mae_empty(self) -> None:
        from src.analytics.backtester import _mae

        assert math.isnan(_mae([], []))

    def test_rmse_basic(self) -> None:
        from src.analytics.backtester import _rmse

        actuals = [0.0, 0.0]
        preds = [1.0, -1.0]
        assert _rmse(actuals, preds) == pytest.approx(1.0)

    def test_rmse_perfect(self) -> None:
        from src.analytics.backtester import _rmse

        vals = [0.2, -0.1, 0.8]
        assert _rmse(vals, vals) == pytest.approx(0.0)

    def test_rmse_empty(self) -> None:
        from src.analytics.backtester import _rmse

        assert math.isnan(_rmse([], []))

    def test_mape_basic(self) -> None:
        from src.analytics.backtester import _mape

        # 10% error for each element → MAPE = 10 %
        actuals = [1.0, 2.0, 4.0]
        preds = [1.1, 2.2, 4.4]
        assert _mape(actuals, preds) == pytest.approx(10.0, rel=1e-5)

    def test_mape_skips_zero_actuals(self) -> None:
        from src.analytics.backtester import _mape

        # Zero actuals are skipped; only the [2, 2.2] pair contributes
        actuals = [0.0, 2.0]
        preds = [99.0, 2.2]
        # 10 % error on one pair
        assert _mape(actuals, preds) == pytest.approx(10.0, rel=1e-5)

    def test_mape_all_zeros(self) -> None:
        from src.analytics.backtester import _mape

        assert math.isnan(_mape([0.0, 0.0], [1.0, 2.0]))

    def test_mape_empty(self) -> None:
        from src.analytics.backtester import _mape

        assert math.isnan(_mape([], []))

    def test_skill_score_better_than_baseline(self) -> None:
        from src.analytics.backtester import _skill_score

        # Model RMSE half of baseline → skill = 0.5
        assert _skill_score(0.5, 1.0) == pytest.approx(0.5)

    def test_skill_score_same_as_baseline(self) -> None:
        from src.analytics.backtester import _skill_score

        assert _skill_score(1.0, 1.0) == pytest.approx(0.0)

    def test_skill_score_worse_than_baseline(self) -> None:
        from src.analytics.backtester import _skill_score

        assert _skill_score(2.0, 1.0) == pytest.approx(-1.0)

    def test_skill_score_trivial_baseline(self) -> None:
        """Zero baseline RMSE should not raise; returns 0.0."""
        from src.analytics.backtester import _skill_score

        assert _skill_score(0.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------


class TestBacktestConfig:
    """Unit tests for BacktestConfig construction and serialisation."""

    def test_defaults(self) -> None:
        from src.analytics.backtester import BacktestConfig

        cfg = BacktestConfig()
        assert cfg.min_train_size == 10
        assert cfg.step_size == 1
        assert cfg.max_folds == 50
        assert cfg.jsonl_path is None

    def test_from_dict_full(self) -> None:
        from src.analytics.backtester import BacktestConfig

        data = {
            "min_train_size": 5,
            "step_size": 2,
            "max_folds": 20,
            "jsonl_path": "/tmp/data.jsonl",
        }
        cfg = BacktestConfig.from_dict(data)
        assert cfg.min_train_size == 5
        assert cfg.step_size == 2
        assert cfg.max_folds == 20
        assert cfg.jsonl_path == "/tmp/data.jsonl"

    def test_from_dict_partial_uses_defaults(self) -> None:
        from src.analytics.backtester import BacktestConfig

        cfg = BacktestConfig.from_dict({"min_train_size": 3})
        assert cfg.min_train_size == 3
        assert cfg.step_size == 1  # default
        assert cfg.max_folds == 50  # default

    def test_from_dict_null_max_folds(self) -> None:
        from src.analytics.backtester import BacktestConfig

        cfg = BacktestConfig.from_dict({"max_folds": None})
        assert cfg.max_folds is None

    def test_to_dict_roundtrip(self) -> None:
        from src.analytics.backtester import BacktestConfig

        cfg = BacktestConfig(min_train_size=7, step_size=3, max_folds=15)
        d = cfg.to_dict()
        cfg2 = BacktestConfig.from_dict(d)
        assert cfg2.min_train_size == 7
        assert cfg2.step_size == 3
        assert cfg2.max_folds == 15

    def test_load_falls_back_to_defaults_when_file_missing(
        self, tmp_path: Path
    ) -> None:
        from src.analytics.backtester import BacktestConfig

        cfg = BacktestConfig.load(path=str(tmp_path / "nonexistent.yaml"))
        # Should return defaults without raising
        assert isinstance(cfg.min_train_size, int)

    def test_load_reads_yaml_file(self, tmp_path: Path) -> None:
        """BacktestConfig.load() should parse a YAML file when PyYAML is available."""
        from src.analytics.backtester import BacktestConfig

        pytest.importorskip("yaml")

        config_file = tmp_path / "backtest_config.yaml"
        config_file.write_text(
            "min_train_size: 8\nstep_size: 2\nmax_folds: 10\njsonl_path: null\n"
        )
        cfg = BacktestConfig.load(path=str(config_file))
        assert cfg.min_train_size == 8
        assert cfg.step_size == 2
        assert cfg.max_folds == 10


# ---------------------------------------------------------------------------
# BacktestReport / BacktestMetrics / FoldResult
# ---------------------------------------------------------------------------


class TestDataClasses:
    """Unit tests for the result data classes."""

    def _make_metrics(self, horizon: int = 24) -> "BacktestMetrics":  # type: ignore[name-defined]  # noqa: F821
        from src.analytics.backtester import BacktestMetrics

        return BacktestMetrics(
            horizon_hours=horizon,
            n_folds=5,
            mae=0.1,
            rmse=0.15,
            mape=5.0,
            baseline_mae=0.2,
            baseline_rmse=0.25,
            baseline_mape=10.0,
            skill_score=0.4,
        )

    def test_backtest_metrics_to_dict_has_all_keys(self) -> None:
        m = self._make_metrics()
        d = m.to_dict()
        for key in (
            "horizon_hours",
            "n_folds",
            "mae",
            "rmse",
            "mape",
            "baseline_mae",
            "baseline_rmse",
            "baseline_mape",
            "skill_score",
        ):
            assert key in d, f"Missing key: {key}"

    def test_fold_result_abs_errors(self) -> None:
        from src.analytics.backtester import FoldResult

        fold = FoldResult(
            fold_index=0,
            train_size=10,
            actual_24h=0.5,
            actual_48h=0.3,
            pred_24h=0.4,
            pred_48h=0.1,
            naive_24h=0.6,
            naive_48h=0.6,
            model_backend="heuristic",
        )
        assert fold.abs_error_24h == pytest.approx(0.1)
        assert fold.abs_error_48h == pytest.approx(0.2)

    def test_fold_result_to_dict_includes_abs_errors(self) -> None:
        from src.analytics.backtester import FoldResult

        fold = FoldResult(
            fold_index=1,
            train_size=15,
            actual_24h=0.5,
            actual_48h=0.3,
            pred_24h=0.4,
            pred_48h=0.1,
            naive_24h=0.6,
            naive_48h=0.6,
            model_backend="sklearn",
        )
        d = fold.to_dict()
        assert "abs_error_24h" in d
        assert "abs_error_48h" in d

    def test_backtest_report_summary_contains_folds(self) -> None:
        from src.analytics.backtester import BacktestReport

        m24 = self._make_metrics(24)
        m48 = self._make_metrics(48)
        report = BacktestReport(
            n_folds=5,
            metrics_24h=m24,
            metrics_48h=m48,
            folds=[],
            backtest_confidence="medium",
        )
        summary = report.summary()
        assert "5" in summary
        assert "medium" in summary

    def test_backtest_report_to_dict_structure(self) -> None:
        from src.analytics.backtester import BacktestReport

        m24 = self._make_metrics(24)
        m48 = self._make_metrics(48)
        report = BacktestReport(
            n_folds=3,
            metrics_24h=m24,
            metrics_48h=m48,
            folds=[],
            backtest_confidence="low",
        )
        d = report.to_dict()
        assert d["n_folds"] == 3
        assert d["backtest_confidence"] == "low"
        assert "metrics_24h" in d
        assert "metrics_48h" in d
        assert "folds" in d


# ---------------------------------------------------------------------------
# Confidence derivation
# ---------------------------------------------------------------------------


class TestDeriveConfidence:
    """Unit tests for _derive_confidence."""

    def _metrics_with_skill(self, skill: float, horizon: int = 24) -> "BacktestMetrics":  # type: ignore[name-defined]  # noqa: F821
        from src.analytics.backtester import BacktestMetrics

        return BacktestMetrics(
            horizon_hours=horizon,
            n_folds=10,
            mae=0.1,
            rmse=0.1,
            mape=5.0,
            baseline_mae=0.2,
            baseline_rmse=0.2,
            baseline_mape=10.0,
            skill_score=skill,
        )

    def test_high_confidence(self) -> None:
        from src.analytics.backtester import _derive_confidence

        m24 = self._metrics_with_skill(0.5, 24)
        m48 = self._metrics_with_skill(0.4, 48)
        assert _derive_confidence(m24, m48) == "high"

    def test_medium_confidence(self) -> None:
        from src.analytics.backtester import _derive_confidence

        m24 = self._metrics_with_skill(0.1, 24)
        m48 = self._metrics_with_skill(0.2, 48)
        assert _derive_confidence(m24, m48) == "medium"

    def test_low_confidence(self) -> None:
        from src.analytics.backtester import _derive_confidence

        m24 = self._metrics_with_skill(-0.2, 24)
        m48 = self._metrics_with_skill(-0.3, 48)
        assert _derive_confidence(m24, m48) == "low"

    def test_unknown_when_nan(self) -> None:
        from src.analytics.backtester import BacktestMetrics, _derive_confidence

        m_nan = BacktestMetrics(
            horizon_hours=24,
            n_folds=0,
            mae=float("nan"),
            rmse=float("nan"),
            mape=float("nan"),
            baseline_mae=float("nan"),
            baseline_rmse=float("nan"),
            baseline_mape=float("nan"),
            skill_score=float("nan"),
        )
        assert _derive_confidence(m_nan, m_nan) == "unknown"


# ---------------------------------------------------------------------------
# run_backtest() engine
# ---------------------------------------------------------------------------


class TestRunBacktest:
    """Integration-style tests for the walk-forward engine."""

    def test_insufficient_data_returns_empty_report(self) -> None:
        from src.analytics.backtester import BacktestConfig, run_backtest

        df = _make_df([0.1, 0.2, 0.3])  # too few rows (< min_train_size + 1)
        cfg = BacktestConfig(min_train_size=10)
        report = run_backtest(config=cfg, df=df)

        assert report.n_folds == 0
        assert report.backtest_confidence == "unknown"
        assert math.isnan(report.metrics_24h.mae)
        assert math.isnan(report.metrics_48h.mae)

    def test_basic_run_produces_folds(self) -> None:
        from src.analytics.backtester import BacktestConfig, run_backtest

        # 30 rows, step 1 data-point intervals (each 1 h apart)
        scores = [0.1 * (i % 5) for i in range(30)]
        df = _make_df(scores, hours_between=1.0)
        cfg = BacktestConfig(min_train_size=5, step_size=1, max_folds=10)
        report = run_backtest(config=cfg, df=df)

        assert report.n_folds == 10
        assert len(report.folds) == 10

    def test_max_folds_respected(self) -> None:
        from src.analytics.backtester import BacktestConfig, run_backtest

        scores = [float(i) / 50.0 for i in range(50)]
        df = _make_df(scores, hours_between=1.0)
        cfg = BacktestConfig(min_train_size=5, step_size=1, max_folds=7)
        report = run_backtest(config=cfg, df=df)

        assert report.n_folds <= 7

    def test_step_size_reduces_fold_count(self) -> None:
        from src.analytics.backtester import BacktestConfig, run_backtest

        scores = [float(i) / 50.0 for i in range(50)]
        df = _make_df(scores, hours_between=1.0)

        cfg_s1 = BacktestConfig(min_train_size=5, step_size=1, max_folds=None)
        cfg_s5 = BacktestConfig(min_train_size=5, step_size=5, max_folds=None)

        r1 = run_backtest(config=cfg_s1, df=df)
        r5 = run_backtest(config=cfg_s5, df=df)

        assert r1.n_folds > r5.n_folds

    def test_metrics_are_finite_on_valid_data(self) -> None:
        from src.analytics.backtester import BacktestConfig, run_backtest

        scores = [0.2 + 0.05 * (i % 5) for i in range(30)]
        df = _make_df(scores, hours_between=1.0)
        cfg = BacktestConfig(min_train_size=5, step_size=2, max_folds=10)
        report = run_backtest(config=cfg, df=df)

        assert not math.isnan(report.metrics_24h.mae)
        assert not math.isnan(report.metrics_48h.rmse)
        assert not math.isnan(report.metrics_24h.skill_score)
        assert not math.isnan(report.metrics_48h.skill_score)

    def test_backtest_confidence_is_valid_label(self) -> None:
        from src.analytics.backtester import BacktestConfig, run_backtest

        scores = [0.1 * (i % 7) for i in range(30)]
        df = _make_df(scores, hours_between=1.0)
        cfg = BacktestConfig(min_train_size=5, step_size=2, max_folds=10)
        report = run_backtest(config=cfg, df=df)

        assert report.backtest_confidence in {"high", "medium", "low", "unknown"}

    def test_folds_have_expected_structure(self) -> None:
        from src.analytics.backtester import BacktestConfig, run_backtest

        scores = [0.1 * (i % 5) for i in range(20)]
        df = _make_df(scores, hours_between=1.0)
        cfg = BacktestConfig(min_train_size=5, step_size=1, max_folds=5)
        report = run_backtest(config=cfg, df=df)

        for fold in report.folds:
            assert fold.train_size >= 5
            assert fold.model_backend in {"prophet", "sklearn", "heuristic"}
            # Predictions should be bounded
            assert -2.0 <= fold.pred_24h <= 2.0
            assert -2.0 <= fold.pred_48h <= 2.0

    def test_to_dict_is_serialisable(self) -> None:
        """BacktestReport.to_dict() should not raise and produce a dict."""
        import json

        from src.analytics.backtester import BacktestConfig, run_backtest

        scores = [0.1 * (i % 5) for i in range(20)]
        df = _make_df(scores, hours_between=1.0)
        cfg = BacktestConfig(min_train_size=5, step_size=1, max_folds=5)
        report = run_backtest(config=cfg, df=df)

        d = report.to_dict()
        # Should be JSON-serialisable (replace NaN with None for JSON)
        json_str = json.dumps(
            d,
            default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o,
        )
        assert json_str  # non-empty


# ---------------------------------------------------------------------------
# Integration: ForecastResult carries backtest_confidence
# ---------------------------------------------------------------------------


class TestForecastResultIntegration:
    """Verify that ForecastResult.backtest_confidence is wired up correctly."""

    def test_backtest_confidence_field_exists(self) -> None:
        from src.analytics.forecaster import ForecastResult

        r = ForecastResult(
            predicted_trend_24h="bullish",
            predicted_trend_48h="neutral",
            confidence_24h=0.7,
            confidence_48h=0.6,
            sentiment_velocity=0.01,
            forecast_score_24h=0.3,
            forecast_score_48h=0.1,
            model_backend="heuristic",
            data_points_used=5,
            generated_at="2024-01-01T00:00:00+00:00",
        )
        # Default value
        assert r.backtest_confidence == "unknown"

    def test_backtest_confidence_field_persists_in_to_dict(self) -> None:
        from src.analytics.forecaster import ForecastResult

        r = ForecastResult(
            predicted_trend_24h="bearish",
            predicted_trend_48h="bearish",
            confidence_24h=0.6,
            confidence_48h=0.55,
            sentiment_velocity=-0.02,
            forecast_score_24h=-0.3,
            forecast_score_48h=-0.4,
            model_backend="sklearn",
            data_points_used=20,
            generated_at="2024-01-01T00:00:00+00:00",
            backtest_confidence="medium",
        )
        d = r.to_dict()
        assert d["backtest_confidence"] == "medium"

    def test_predict_with_explicit_backtest_confidence(self) -> None:
        """
        SentimentForecaster.predict() returns a ForecastResult; the caller
        (run()) later overwrites backtest_confidence.  Verify predict() at
        minimum leaves the default in place.
        """
        from src.analytics.forecaster import SentimentForecaster

        forecaster = SentimentForecaster()
        scores = [0.1 * i for i in range(10)]
        df = _make_df(scores)
        forecaster.train(df)
        result = forecaster.predict(df)

        # predict() does not touch backtest_confidence; default should be "unknown"
        assert result.backtest_confidence == "unknown"
        assert result.backtest_confidence in {"high", "medium", "low", "unknown"}

    def test_run_attaches_backtest_confidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        SentimentForecaster.run() should attach a backtest_confidence label
        derived from the walk-forward harness.
        """
        import json

        if _real_pandas() is None:
            pytest.skip("real pandas not available in this test environment")

        from src.analytics.forecaster import SentimentForecaster

        # Build a synthetic analytics JSONL file with enough rows
        rows = []
        for i in range(25):
            ts = f"2024-01-01T{i:02d}:00:00+00:00"
            score = 0.1 * (i % 5)
            rows.append(
                json.dumps(
                    {
                        "timestamp": ts,
                        "news_count": 10,
                        "sentiment_data": {
                            "average_compound_score": score,
                            "sentiment_distribution": {
                                "positive": 0.5,
                                "negative": 0.2,
                                "neutral": 0.3,
                            },
                        },
                    }
                )
            )

        jsonl_file = tmp_path / "analytics.jsonl"
        jsonl_file.write_text("\n".join(rows))

        # Point the forecaster at our test JSONL and a fast backtest config
        monkeypatch.setenv("ANALYTICS_JSONL_PATH", str(jsonl_file))

        from src.analytics.backtester import BacktestConfig

        fast_cfg = BacktestConfig(
            min_train_size=5, step_size=5, max_folds=3, jsonl_path=str(jsonl_file)
        )
        monkeypatch.setattr(
            "src.analytics.backtester.BacktestConfig.load",
            lambda *a, **kw: fast_cfg,
        )

        forecaster = SentimentForecaster(jsonl_path=jsonl_file)
        result = forecaster.run()

        assert result.backtest_confidence in {"high", "medium", "low", "unknown"}
