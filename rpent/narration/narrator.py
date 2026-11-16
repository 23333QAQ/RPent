# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Narrate new observation frames in a background thread while the planner runs."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rpent.narration.config import NarrationConfig
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.session import EnvState

logger = get_logger("narration")

LOG_NAME = "narration.jsonl"


class Describer(Protocol):
    async def describe(self, image: bytes, focus: str | None = None) -> str: ...

    async def close(self) -> None: ...


class Synthesizer(Protocol):
    async def synthesize(self, text: str) -> str: ...


class Narrator:
    """Describe each new frame and synthesize the descriptions in order.

    One background thread runs its own event loop with two tasks: a watcher
    that checks ``state`` every ``interval_s`` and describes a frame once
    when its step first carries a configured camera image, and a speaker
    that synthesizes queued descriptions one at a time. Failures are logged
    and recorded in ``narration.jsonl``; they never reach the planner thread.
    Control methods are thread-safe and return without waiting for model or
    speech requests, except that restarting waits for a previous run to drain.
    """

    def __init__(
        self,
        config: NarrationConfig,
        *,
        state: EnvState,
        output_dir: str | Path,
        describer_factory: Callable[[], Describer],
        synthesizer_factory: Callable[[], Synthesizer],
    ) -> None:
        self._config = config
        self._state = state
        self._log_path = Path(output_dir) / LOG_NAME
        self._describer_factory = describer_factory
        self._synthesizer_factory = synthesizer_factory
        self._control = threading.Lock()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._interval_s = config.interval_s
        self._focus: str | None = None
        self._counts = {"spoken": 0, "failed": 0, "dropped": 0, "pending": 0}

    @property
    def running(self) -> bool:
        """Whether a narration run is active and not stopping."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and self._stop is not None
            and not self._stop.is_set()
        )

    def start(
        self, *, interval_s: float | None = None, focus: str | None = None
    ) -> dict[str, Any]:
        """Start narrating, or update the interval and focus of a running one.

        Args:
            interval_s: Seconds between frame checks; ``None`` keeps the
                current value.
            focus: What descriptions should emphasize; ``None`` keeps the
                current hint and an empty string clears it.

        Returns:
            The run status and the settings now in effect.
        """
        with self._control:
            with self._lock:
                if interval_s is not None:
                    self._interval_s = interval_s
                if focus is not None:
                    self._focus = focus.strip() or None
            if self.running:
                return self._settings("updated")
            self._join_previous()
            stop = threading.Event()
            thread = threading.Thread(
                target=self._thread_main, args=(stop,), name="narration", daemon=True
            )
            self._stop, self._thread = stop, thread
            thread.start()
            logger.info("Narration started (interval %.1fs)", self._interval_s)
            return self._settings("running")

    def stop(self) -> dict[str, Any]:
        """Stop checking for frames; queued text is still synthesized.

        Returns:
            ``stopping`` with the remaining queue and counts, or ``stopped``
            when no run was active.
        """
        with self._control:
            if not self.running:
                return {"status": "stopped", **self._snapshot()}
            assert self._stop is not None
            self._stop.set()
            logger.info("Narration stopping")
            return {"status": "stopping", **self._snapshot()}

    def close(self) -> None:
        """Stop and wait for the background thread to finish draining."""
        with self._control:
            if self._stop is not None:
                self._stop.set()
            self._join_previous()

    # -- background thread ------------------------------------------------

    def _join_previous(self) -> None:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        timeout = self._config.request_timeout_s + self._config.drain_timeout_s + 5
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning("Narration thread did not stop within %.0fs", timeout)

    def _thread_main(self, stop: threading.Event) -> None:
        try:
            asyncio.run(self._run(stop))
        except Exception:
            logger.exception("Narration stopped after an unexpected error")
        finally:
            stop.set()

    async def _run(self, stop: threading.Event) -> None:
        describer = self._describer_factory()
        synthesizer = self._synthesizer_factory()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._config.max_queue
        )
        speaking = asyncio.create_task(self._speak(queue, synthesizer))
        try:
            await self._watch(queue, describer, stop)
            try:
                await asyncio.wait_for(queue.join(), self._config.drain_timeout_s)
            except asyncio.TimeoutError:
                logger.warning(
                    "Narration dropped %d queued sentence(s) at shutdown",
                    queue.qsize(),
                )
        finally:
            speaking.cancel()
            await asyncio.gather(speaking, return_exceptions=True)
            await describer.close()
            with self._lock:
                self._counts["pending"] = 0

    async def _watch(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        describer: Describer,
        stop: threading.Event,
    ) -> None:
        last_step: int | None = None
        while not stop.is_set():
            frame = self._latest_frame()
            if frame is not None and frame[0] != last_step:
                last_step = frame[0]
                await self._describe(queue, describer, *frame)
            with self._lock:
                interval_s = self._interval_s
            if await asyncio.to_thread(stop.wait, interval_s):
                return

    def _latest_frame(self) -> tuple[int, str] | None:
        """Return the newest step and camera whose image is fully saved."""
        try:
            record = self._state.get(-1)
        except (LookupError, KeyError):
            return None
        for camera in self._config.cameras:
            # EnvState adds an artifact name only after its file is in place.
            if camera in record.artifacts:
                return record.step_idx, camera
        return None

    async def _describe(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        describer: Describer,
        step: int,
        camera: str,
    ) -> None:
        detected = time.time()
        try:
            image = await asyncio.to_thread(self._state.load_bytes, camera, step=step)
        except OSError as exc:
            self._record(step=step, camera=camera, error=f"read: {exc}")
            return
        with self._lock:
            focus = self._focus
        started = time.perf_counter()
        try:
            text = await asyncio.wait_for(
                describer.describe(image, focus), self._config.request_timeout_s
            )
        except Exception as exc:
            self._record(
                step=step, camera=camera, error=f"describe: {_error_text(exc)}"
            )
            return
        item = {
            "step": step,
            "camera": camera,
            "text": text,
            "describe_s": round(time.perf_counter() - started, 3),
            "detected": detected,
        }
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            self._record(**item, error="queue full; sentence dropped", dropped=True)
            return
        with self._lock:
            self._counts["pending"] += 1

    async def _speak(
        self, queue: asyncio.Queue[dict[str, Any]], synthesizer: Synthesizer
    ) -> None:
        while True:
            item = await queue.get()
            started = time.perf_counter()
            try:
                audio_path = await asyncio.wait_for(
                    synthesizer.synthesize(item["text"]),
                    self._config.synthesis_timeout_s,
                )
            except Exception as exc:
                self._record(**item, error=f"synthesize: {_error_text(exc)}")
            else:
                self._record(
                    **item,
                    audio_path=audio_path,
                    synthesize_s=round(time.perf_counter() - started, 3),
                )
            finally:
                with self._lock:
                    self._counts["pending"] -= 1
                queue.task_done()

    # -- bookkeeping ------------------------------------------------------

    def _record(self, *, dropped: bool = False, **entry: Any) -> None:
        detected = entry.pop("detected", None)
        if detected is not None:
            entry["lag_s"] = round(time.time() - detected, 3)
        error = entry.get("error")
        with self._lock:
            key = "dropped" if dropped else "failed" if error else "spoken"
            self._counts[key] += 1
            try:
                with self._log_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("Could not write %s: %s", self._log_path, exc)
        if error:
            logger.warning("Narration step %s: %s", entry.get("step"), error)
        else:
            logger.info(
                "Narration step %s: %s -> %s",
                entry.get("step"),
                entry.get("text"),
                entry.get("audio_path"),
            )

    def _snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def _settings(self, status: str) -> dict[str, Any]:
        with self._lock:
            return {
                "status": status,
                "interval_s": self._interval_s,
                "focus": self._focus,
                "log": str(self._log_path),
            }


def _error_text(exc: BaseException) -> str:
    detail = str(exc) or type(exc).__name__
    return f"{type(exc).__name__}: {detail}"[:300]
