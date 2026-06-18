"""Registry for named retrieval pipelines."""

from __future__ import annotations

from services.retrieval_strategies.advanced import AdvancedPipeline
from services.retrieval_strategies.basic import BasicPipeline
from services.retrieval_strategies.hybrid import HybridPipeline

DEFAULT_RETRIEVAL_STRATEGY = "hybrid"


def _strategies():
    return {
        "basic": BasicPipeline(),
        "hybrid": HybridPipeline(),
        "advanced": AdvancedPipeline(),
    }


def list_retrieval_strategies() -> list[str]:
    return sorted(_strategies().keys())


def get_retrieval_strategy(name: str | None):
    strategies = _strategies()
    strategy_name = (name or DEFAULT_RETRIEVAL_STRATEGY).lower()
    if strategy_name not in strategies:
        valid = ", ".join(list_retrieval_strategies())
        raise ValueError(
            f"Unknown retrieval strategy '{name}'. Valid strategies: {valid}"
        )
    return strategies[strategy_name]
