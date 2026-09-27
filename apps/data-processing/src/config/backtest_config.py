# -*- coding: utf-8 -*-
"""
Configuration loader for the backtesting harness.

Re-exports :class:`BacktestConfig` from :mod:`src.analytics.backtester` so
that other modules can import it from the conventional ``src.config`` package
without creating a circular dependency.

Usage::

    from src.config.backtest_config import BacktestConfig

    cfg = BacktestConfig.load()          # from config/backtest_config.yaml
    cfg = BacktestConfig.load("my.yaml") # explicit path
"""

from src.analytics.backtester import BacktestConfig

__all__ = ["BacktestConfig"]
