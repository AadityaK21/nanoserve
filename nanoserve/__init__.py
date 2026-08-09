"""nanoserve -- a small LLM inference server built from scratch.

Public surface:

    from nanoserve import LLMEngine, EngineConfig, SamplingParams

Everything below that is deliberately importable on its own so the pieces can
be tested in isolation: block_manager and scheduler have no torch dependency at
all.
"""

from .config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SamplingParams,
    SchedulerConfig,
)

__all__ = [
    "CacheConfig",
    "EngineConfig",
    "ModelConfig",
    "SamplingParams",
    "SchedulerConfig",
    "LLMEngine",
]

__version__ = "0.5.0"


def __getattr__(name):
    # Lazy so that `import nanoserve` works without torch installed, which
    # keeps the pure-python tests runnable anywhere.
    if name == "LLMEngine":
        from .engine import LLMEngine

        return LLMEngine
    raise AttributeError(name)
