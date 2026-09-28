"""Laya GPU inference service — Python port of @receptron/laya's request rendering + ONNX worker."""

__version__ = "0.1.0"

# Lazy imports to avoid triggering onnxruntime/tokenizers initialization at package level.
# Direct imports of heavy modules (onnxruntime, tokenizers) cause issues in test
# environments and CI where GPU is unavailable.

def __getattr__(name: str):
    if name == "LayaWorker":
        from .worker import LayaWorker
        return LayaWorker
    if name == "RequestBatcher":
        from .batcher import RequestBatcher
        return RequestBatcher
    if name == "ResultCache":
        from .result_cache import ResultCache
        return ResultCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "LayaWorker",
    "RequestBatcher",
    "ResultCache",
]
