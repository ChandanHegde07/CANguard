"""Pluggable window representations (residualization strategies) for PIRD."""

from .strategies import (
    DEFAULT_ORDER,
    RESIDUALIZERS,
    BaseResidualizer,
    GlobalZResidualizer,
    PerIdMADResidualizer,
    PerIdQuantileResidualizer,
    PerIdZResidualizer,
    RawResidualizer,
    build_residualizer,
    list_residualizers,
)

__all__ = [
    "DEFAULT_ORDER",
    "RESIDUALIZERS",
    "BaseResidualizer",
    "GlobalZResidualizer",
    "PerIdMADResidualizer",
    "PerIdQuantileResidualizer",
    "PerIdZResidualizer",
    "RawResidualizer",
    "build_residualizer",
    "list_residualizers",
]
