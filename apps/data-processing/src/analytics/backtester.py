# -*- coding: utf-8 -*-
"""
Walk-forward backtesting harness for :class:`SentimentForecaster`.

The harness slices historical data into an expanding training window and a
single-step test window, trains a fresh :class:`SentimentForecaster` on each
fold, records predictions against actuals, then aggregates error metrics and
compares against a naïve (last-value) baseline.

Typical usage::

    from src.analytics.backtester import run_backtest, BacktestConfig
    from src.analytics.forecaster import SentimentForecaster

    config = BacktestConfig.load()
    results = run_backtest(config)
    print(results.summary())

All parameters are drawn from :class:`BacktestConfig` so results are
deterministic given the same config file and input data.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _mae(actuals: List[float], preds: List[float]) -> float:
    """Mean Absolute Error."""
    if not actuals:
        return float("nan")
    return sum(abs(a - p) for a, p in zip(actuals, preds)) / len(actuals)


def _rmse(actuals: List[float], preds: List[float]) -> float:
    """Root Mean Squared Error."""
    if not actuals:
        return float("nan")
    mse = sum((a - p) ** 2 for a, p in zip(actuals, preds)) / len(actuals)
    return math.sqrt(mse)


def _mape(actuals: List[float], preds: List[float]) -> float:
    """Mean Absolute Percentage Error (skips zeros in actuals to avoid division by zero)."""
    if not actuals:
        return float("nan")
    pairs = [(a, p) for a, p in zip(actuals, preds) if abs(a) > 1e-9]
    if not pairs:
        return float("nan")
    return sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs) * 100.0


def _skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """
    Skill score = 1 − (model_rmse / baseline_rmse).

    Positive → model beats baseline.  0 → same as baseline.  Negative → worse.
    Returns 0.0 when baseline_rmse is zero (trivial problem).
    """
    if baseline_rmse < 1e-12:
        return 0.0
    return float(1.0 - model_rmse / baseline_rmse)


# ---------------------------------------------------------------------------
# Per-fold result
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold_index: int
    train_size: int
    #: Actual sentiment score at the target horizon
    actual_24h: float
    actual_48h: float
    #: Model predictions
    pred_24h: float
    pred_48h: float
    #: Naïve baseline (last observed value carried forward)
    naive_24h: float
    naive_48h: float
    model_backend: str

    # Derived convenience properties ─────────────────────────────────────────

    @property
    def abs_error_24h(self) -> float:
        return abs(self.actual_24h - self.pred_24h)

    @property
    def abs_error_48h(self) -> float:
        return abs(self.actual_48h - self.pred_48h)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["abs_error_24h"] = self.abs_error_24h
        d["abs_error_48h"] = self.abs_error_48h
        return d


# ---------------------------------------------------------------------------
# Aggregate backtest results
# ---------------------------------------------------------------------------


@dataclass
class BacktestMetrics:
    """Error metrics aggregated across all folds for one forecast horizon."""

    horizon_hours: int
    n_folds: int
    mae: float
    rmse: float
    mape: float
    #: Naïve baseline metrics
    baseline_mae: float
    baseline_rmse: float
    baseline_mape: float
    #: Skill score vs naïve baseline (positive = better than baseline)
    skill_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestReport:
    """Full walk-forward backtest report."""

    n_folds: int
    metrics_24h: BacktestMetrics
    metrics_48h: BacktestMetrics
    folds: List[FoldResult] = field(default_factory=list)
    #: Derived confidence label for use in API responses
    backtest_confidence: str = "unknown"

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Walk-forward backtest  ({self.n_folds} folds)",
            "",
            "  Horizon  | Metric  | Model  | Baseline | Skill",
            "  ---------|---------|--------|----------|------",
        ]
        for m in (self.metrics_24h, self.metrics_48h):
            lines.append(
                f"  {m.horizon_hours:>5}h   | MAE     | {m.mae:6.4f} | {m.baseline_mae:8.4f} |"
                f" {m.skill_score:+.3f}"
            )
            lines.append(
                f"           | RMSE    | {m.rmse:6.4f} | {m.baseline_rmse:8.4f} |"
            )
            lines.append(
                f"           | MAPE    | {m.mape:6.2f}% | {m.baseline_mape:8.2f}% |"
            )
        lines.append("")
        lines.append(f"  Backtest confidence: {self.backtest_confidence}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "backtest_confidence": self.backtest_confidence,
            "metrics_24h": self.metrics_24h.to_dict(),
            "metrics_48h": self.metrics_48h.to_dict(),
            "folds": [f.to_dict() for f in self.folds],
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """
    Reproducible configuration for the walk-forward backtest.

    Loaded from ``config/backtest_config.yaml`` by default so that any run
    with the same committed config file produces identical fold splits and
    model hyper-parameters.
    """

    #: Minimum number of rows required in the training window before the first
    #: prediction is made.
    min_train_size: int = 10
    #: Number of rows to advance the training window by between folds.  Use 1
    #: for a pure expanding-window walk-forward test.
    step_size: int = 1
    #: Maximum number of folds to evaluate (None → unlimited).
    max_folds: Optional[int] = 50
    #: Path to analytics JSONL input (None → use forecaster default).
    jsonl_path: Optional[str] = None
    #: Where to look for the YAML config file.
    config_path: str = "config/backtest_config.yaml"

    # ── Loaders ───────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
    ) -> "BacktestConfig":
        """
        Load config from a YAML file.  Falls back to defaults when the file
        is absent or when PyYAML is not installed.
        """
        config_path = path or "config/backtest_config.yaml"
        p = Path(config_path)
        if not p.exists():
            logger.warning(
                f"Backtest config not found at {config_path}; using defaults"
            )
            return cls(config_path=config_path)

        try:
            import yaml  # type: ignore

            with open(p) as fh:
                data: Dict[str, Any] = yaml.safe_load(fh) or {}
            return cls.from_dict(data, config_path=str(p))
        except ImportError:
            logger.warning("PyYAML not installed; using default BacktestConfig")
            return cls(config_path=config_path)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], config_path: str = "config/backtest_config.yaml"
    ) -> "BacktestConfig":
        """Construct from a plain dictionary (e.g. parsed YAML)."""
        # Distinguish explicit null (None) from absent key (use default 50)
        _sentinel = object()
        raw_max_folds = data.get("max_folds", _sentinel)
        if raw_max_folds is _sentinel:
            max_folds: Optional[int] = 50
        elif raw_max_folds is None:
            max_folds = None
        else:
            max_folds = int(raw_max_folds)

        return cls(
            min_train_size=int(data.get("min_train_size", 10)),
            step_size=int(data.get("step_size", 1)),
            max_folds=max_folds,
            jsonl_path=data.get("jsonl_path"),
            config_path=config_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Confidence derivation
# ---------------------------------------------------------------------------

_CONFIDENCE_THRESHOLDS: List[Tuple[float, str]] = [
    # skill_score threshold → label
    (0.30, "high"),
    (0.05, "medium"),
    (-1.0, "low"),
]


def _derive_confidence(
    metrics_24h: BacktestMetrics, metrics_48h: BacktestMetrics
) -> str:
    """
    Translate backtest skill scores into a human-readable confidence label.

    The combined skill score is the average of the 24 h and 48 h skill
    scores.  A positive skill score means the model beats the naïve baseline.
    """
    if math.isnan(metrics_24h.skill_score) or math.isnan(metrics_48h.skill_score):
        return "unknown"

    combined = (metrics_24h.skill_score + metrics_48h.skill_score) / 2.0
    for threshold, label in _CONFIDENCE_THRESHOLDS:
        if combined >= threshold:
            return label
    return "low"


# ---------------------------------------------------------------------------
# Core walk-forward engine
# ---------------------------------------------------------------------------


def _build_fold_metrics(
    horizon_hours: int,
    actuals: List[float],
    preds: List[float],
    naive_preds: List[float],
) -> BacktestMetrics:
    model_rmse = _rmse(actuals, preds)
    baseline_rmse = _rmse(actuals, naive_preds)
    return BacktestMetrics(
        horizon_hours=horizon_hours,
        n_folds=len(actuals),
        mae=_mae(actuals, preds),
        rmse=model_rmse,
        mape=_mape(actuals, preds),
        baseline_mae=_mae(actuals, naive_preds),
        baseline_rmse=baseline_rmse,
        baseline_mape=_mape(actuals, naive_preds),
        skill_score=_skill_score(model_rmse, baseline_rmse),
    )


def run_backtest(
    config: Optional[BacktestConfig] = None,
    df: Optional[pd.DataFrame] = None,
) -> BacktestReport:
    """
    Run a walk-forward backtest of :class:`SentimentForecaster`.

    Parameters
    ----------
    config:
        Backtest configuration.  Defaults to ``BacktestConfig.load()``.
    df:
        Pre-loaded DataFrame (as returned by
        :meth:`SentimentForecaster.load_history`).  When *None*, data is
        loaded from the path specified in *config*.

    Returns
    -------
    BacktestReport
        Aggregated metrics plus per-fold detail.
    """
    from src.analytics.forecaster import SentimentForecaster

    if config is None:
        config = BacktestConfig.load()

    forecaster_template = SentimentForecaster(
        jsonl_path=Path(config.jsonl_path) if config.jsonl_path else None
    )

    if df is None:
        df = forecaster_template.load_history()

    n = len(df)

    if n < config.min_train_size + 1:
        logger.warning(
            f"Not enough data for backtesting "
            f"(need >= {config.min_train_size + 1} rows, got {n})"
        )
        empty_metrics = BacktestMetrics(
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
        return BacktestReport(
            n_folds=0,
            metrics_24h=empty_metrics,
            metrics_48h=BacktestMetrics(
                horizon_hours=48,
                n_folds=0,
                mae=float("nan"),
                rmse=float("nan"),
                mape=float("nan"),
                baseline_mae=float("nan"),
                baseline_rmse=float("nan"),
                baseline_mape=float("nan"),
                skill_score=float("nan"),
            ),
            folds=[],
            backtest_confidence="unknown",
        )

    # Estimate how many rows correspond to ~24 h and ~48 h
    if n >= 2:
        median_h = float(
            df["timestamp"].diff().dropna().dt.total_seconds().median() / 3600.0
        )
        median_h = max(median_h, 0.01)
    else:
        median_h = 1.0

    step_24h = max(1, round(24.0 / median_h))
    step_48h = max(1, round(48.0 / median_h))

    actuals_24h: List[float] = []
    actuals_48h: List[float] = []
    preds_24h: List[float] = []
    preds_48h: List[float] = []
    naive_24h: List[float] = []
    naive_48h: List[float] = []
    fold_results: List[FoldResult] = []

    fold_index = 0
    train_end = config.min_train_size  # exclusive upper bound of training slice

    while train_end < n:
        # Determine test targets
        target_idx_24h = min(train_end + step_24h - 1, n - 1)
        target_idx_48h = min(train_end + step_48h - 1, n - 1)

        actual_24h = float(df["sentiment_score"].iloc[target_idx_24h])
        actual_48h = float(df["sentiment_score"].iloc[target_idx_48h])

        train_df = df.iloc[:train_end].copy()

        # Naïve baseline: last observed value carried forward
        naive = float(train_df["sentiment_score"].iloc[-1])

        # Train a fresh forecaster on training slice only
        f = SentimentForecaster(
            jsonl_path=Path(config.jsonl_path) if config.jsonl_path else None
        )
        f.train(train_df)
        result = f.predict(train_df)

        pred_24h = result.forecast_score_24h
        pred_48h = result.forecast_score_48h

        actuals_24h.append(actual_24h)
        actuals_48h.append(actual_48h)
        preds_24h.append(pred_24h)
        preds_48h.append(pred_48h)
        naive_24h.append(naive)
        naive_48h.append(naive)

        fold_results.append(
            FoldResult(
                fold_index=fold_index,
                train_size=len(train_df),
                actual_24h=actual_24h,
                actual_48h=actual_48h,
                pred_24h=pred_24h,
                pred_48h=pred_48h,
                naive_24h=naive,
                naive_48h=naive,
                model_backend=result.model_backend,
            )
        )

        fold_index += 1
        if config.max_folds is not None and fold_index >= config.max_folds:
            break

        train_end += config.step_size

    logger.info(f"Backtest complete: {fold_index} folds evaluated")

    m24 = _build_fold_metrics(24, actuals_24h, preds_24h, naive_24h)
    m48 = _build_fold_metrics(48, actuals_48h, preds_48h, naive_48h)
    confidence = _derive_confidence(m24, m48)

    return BacktestReport(
        n_folds=fold_index,
        metrics_24h=m24,
        metrics_48h=m48,
        folds=fold_results,
        backtest_confidence=confidence,
    )
