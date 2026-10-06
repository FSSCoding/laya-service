"""
RequestBatcher — coalesce concurrent HTTP requests into one GPU forward pass.

The idea is the same as Laya's own batching (multiple questions -> one call): a burst of
concurrent requests gets merged into one onnxruntime run instead of many serialised runs.

Flow:
  caller -> submit(requests) -> puts (requests, Future) in queue -> awaits Future
  _worker drains queue every MAX_WAIT_MS -> merges all requests -> one worker call
           -> resolves each Future with its result
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("laya-service.batcher")


class RequestBatcher:
    def __init__(
        self, max_batch_size: int = 512, max_wait_ms: float = 5.0,
        worker: Optional[Callable] = None,
        worker_many: Optional[Callable] = None,
        on_coalesce: Optional[Callable] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._worker_fn = worker
        self._worker_fn_many = worker_many
        self._on_coalesce = on_coalesce  # called with batch_size when coalescing happens
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._queue = asyncio.Queue()
        self._task = loop.create_task(self._drain())
        logger.info(
            f"RequestBatcher started: max_batch={self.max_batch_size} wait={self.max_wait_ms}ms"
        )

    async def submit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((payload, future))
        return await future

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                first_payload, first_future = await self._queue.get()
            except asyncio.CancelledError:
                return

            batch: List[Tuple[Dict[str, Any], asyncio.Future]] = [(first_payload, first_future)]

            # Drain additional items that arrived during the wait window
            deadline = loop.time() + self.max_wait_ms / 1000.0
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    payload, future = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append((payload, future))
                except asyncio.TimeoutError:
                    break

            # Run the blocking worker call in the default thread executor so the event loop stays
            # free.
            if len(batch) > 1:
                logger.debug(f"Coalesced {len(batch)} requests into one GPU call")
                if self._on_coalesce:
                    self._on_coalesce(len(batch))
            try:
                if self._worker_fn_many is not None and len(batch) > 1:
                    # Real coalescing: merge all payloads into one forward pass.
                    pairs = [(p["state"], p["questions"]) for p, _ in batch]
                    results = await loop.run_in_executor(
                        None, self._worker_fn_many, pairs
                    )
                else:
                    results = await loop.run_in_executor(
                        None, self._process_batch, [p for p, _ in batch]
                    )
                for (payload, future), res in zip(batch, results):
                    if not future.done():
                        future.set_result(res)
            except Exception as exc:  # noqa: BLE001 - propagate to all waiters
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)

    def _process_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Blocking: run the shared worker over each payload."""
        out: List[Dict[str, Any]] = []
        for p in payloads:
            out.append(self._worker_fn(p))
        return out
