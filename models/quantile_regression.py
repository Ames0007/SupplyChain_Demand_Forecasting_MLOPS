"""
Quantile regression module — re-exports QuantileLightGBM for explicit import path.

The primary implementation lives in models/tree_models.py.
This module exists as a named entry point matching the project structure spec.
"""

from models.tree_models import QuantileLightGBM  # noqa: F401

__all__ = ["QuantileLightGBM"]
