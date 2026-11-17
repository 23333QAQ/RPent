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

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.narration import narration_session
from rpent.narration.config import NarrationConfig
from rpent.narration.narrator import LOG_NAME, Narrator
from rpent.session import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import init_output_dir
from rpent.utils.mcp_servers import McpServerSpec

pytest.importorskip("PIL.Image")

CAMERA = "agentview_high.png"


def make_config(**overrides) -> NarrationConfig:
    values = {
        "tts_server": "tts",
        "model": "vision",
        "base_url": "http://127.0.0.1:1/v1",
        "api_key": "key",
        "interval_s": 0.02,
        "request_timeout_s": 2.0,
        "synthesis_timeout_s": 2.0,
        "drain_timeout_s": 2.0,
        **overrides,
    }
    return NarrationConfig(**values)


def add_frame(state: EnvState, *, image: bool = True) -> int:
    with state.record_step(state={}) as step:
        if image:
            state.save(CAMERA, np.full((4, 4, 3), step, dtype=np.uint8))
    return step


def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


class FakeDescriber:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[tuple[bytes, str | None]] = []
        self.fail_first = fail_first
        self.closed = False

    async def describe(self, image: bytes, focus: str | None = None) -> str:
        self.calls.append((image, focus))
        if self.fail_first and len(self.calls) == 1:
            raise TimeoutError("model timed out")
        return f"sentence {len(self.calls)}"

    async def close(self) -> None:
        self.closed = True


class FakeSynthesizer:
    def __init__(self, fail_text: str | None = None) -> None:
        self.texts: list[str] = []
        self.fail_text = fail_text
        self.release = threading.Event()
        self.release.set()

    async def synthesize(self, text: str) -> str:
        await asyncio.to_thread(self.release.wait, 5)
        if text == self.fail_text:
            raise RuntimeError("tts unavailable")
        self.texts.append(text)
        return f"/audio/{len(self.texts)}.wav"


@pytest.fixture
def state(tmp_path):
    return EnvState(tmp_path / "state")


def make_narrator(state, tmp_path, describer, synthesizer, **config) -> Narrator:
    return Narrator(
        make_config(**config),
        state=state,
        output_dir=tmp_path,
        describer_factory=lambda: describer,
        synthesizer_factory=lambda: synthesizer,
    )


def read_log(tmp_path) -> list[dict]:
    path = tmp_path / LOG_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_each_new_frame_is_described_once_and_spoken_in_order(state, tmp_path):
    describer, synthesizer = FakeDescriber(), FakeSynthesizer()
    narrator = make_narrator(state, tmp_path, describer, synthesizer)
    add_frame(state)

    narrator.start()
    try:
        wait_until(lambda: len(synthesizer.texts) == 1)
        time.sleep(0.1)  # Several checks of an unchanged frame.
        add_frame(state)
        wait_until(lambda: len(synthesizer.texts) == 2)
        add_frame(state)
        wait_until(lambda: len(synthesizer.texts) == 3)
    finally:
        narrator.close()

    assert len(describer.calls) == 3
    assert synthesizer.texts == ["sentence 1", "sentence 2", "sentence 3"]
    assert describer.closed
    log = read_log(tmp_path)
    assert [entry["step"] for entry in log] == [0, 1, 2]
    assert [entry["audio_path"] for entry in log] == [
        "/audio/1.wav",
        "/audio/2.wav",
        "/audio/3.wav",
    ]


def test_step_is_described_only_after_its_image_is_saved(state, tmp_path):
    describer, synthesizer = FakeDescriber(), FakeSynthesizer()
    narrator = make_narrator(state, tmp_path, describer, synthesizer)
    add_frame(state, image=False)

    narrator.start()
    try:
        time.sleep(0.1)
        assert describer.calls == []
        state.save(CAMERA, np.zeros((4, 4, 3), dtype=np.uint8))
        wait_until(lambda: len(synthesizer.texts) == 1)
    finally:
        narrator.close()

    assert read_log(tmp_path)[0]["step"] == 0


def test_failures_are_logged_and_narration_continues(state, tmp_path):
    describer = FakeDescriber(fail_first=True)
    synthesizer = FakeSynthesizer(fail_text="sentence 2")
    narrator = make_narrator(state, tmp_path, describer, synthesizer)
    add_frame(state)

    narrator.start()
    try:
        wait_until(lambda: len(describer.calls) == 1)
        for expected in (2, 3):
            add_frame(state)
            wait_until(lambda n=expected: len(describer.calls) == n)
        wait_until(lambda: synthesizer.texts == ["sentence 3"])
    finally:
        narrator.close()

    log = read_log(tmp_path)
    assert log[0]["error"].startswith("describe: TimeoutError")
    assert log[1]["error"].startswith("synthesize: RuntimeError")
    assert log[2]["audio_path"] == "/audio/1.wav"
    assert narrator.stop() == {
        "status": "stopped",
        "spoken": 1,
        "failed": 2,
        "dropped": 0,
        "pending": 0,
    }


def test_stop_returns_immediately_and_queued_text_is_still_spoken(state, tmp_path):
    describer, synthesizer = FakeDescriber(), FakeSynthesizer()
    synthesizer.release.clear()
    narrator = make_narrator(state, tmp_path, describer, synthesizer)
    add_frame(state)

    narrator.start()
    try:
        wait_until(lambda: len(describer.calls) == 1)
        add_frame(state)
        wait_until(lambda: len(describer.calls) == 2)
        started = time.monotonic()
        result = narrator.stop()
        assert time.monotonic() - started < 0.5
        assert result["status"] == "stopping"
        assert result["pending"] == 2
        synthesizer.release.set()
    finally:
        narrator.close()

    assert synthesizer.texts == ["sentence 1", "sentence 2"]
    assert not narrator.running


def test_start_while_running_updates_interval_and_focus(state, tmp_path):
    describer, synthesizer = FakeDescriber(), FakeSynthesizer()
    narrator = make_narrator(state, tmp_path, describer, synthesizer)

    first = narrator.start()
    try:
        second = narrator.start(interval_s=0.05, focus="the green bottle")
        add_frame(state)
        wait_until(lambda: len(synthesizer.texts) == 1)
    finally:
        narrator.close()

    assert first["status"] == "running"
    assert second["status"] == "updated"
    assert second["interval_s"] == 0.05
    assert describer.calls[0][1] == "the green bottle"


def test_narrator_can_restart_after_stopping(state, tmp_path):
    describer, synthesizer = FakeDescriber(), FakeSynthesizer()
    narrator = make_narrator(state, tmp_path, describer, synthesizer)
    add_frame(state)

    narrator.start()
    wait_until(lambda: len(synthesizer.texts) == 1)
    narrator.stop()
    add_frame(state)
    assert narrator.start()["status"] == "running"
    try:
        wait_until(lambda: len(synthesizer.texts) == 2)
    finally:
        narrator.close()


class MinimalToolkit(Toolkit):
    def __init__(self, state: EnvState) -> None:
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            memory=SimpleNamespace(),
            state=state,
        )

    def _register_common_tools(self) -> None:
        pass


def test_session_without_config_adds_no_tools(state, tmp_path):
    toolkit = MinimalToolkit(state)

    with narration_session(None, toolkit=toolkit, mcp_servers=(), output_dir=tmp_path):
        assert toolkit.list_tools() == ()


def test_session_tools_control_narration_without_blocking(state, tmp_path, monkeypatch):
    import rpent.narration as narration_module
    import rpent.narration.describer as describer_module
    import rpent.narration.speaker as speaker_module

    site = tmp_path / "site.yaml"
    monkeypatch.setattr(
        narration_module,
        "load_narration_config",
        lambda path: make_config() if path == site else None,
    )
    describer, synthesizer = FakeDescriber(), FakeSynthesizer()
    monkeypatch.setattr(describer_module, "SceneDescriber", lambda config: describer)
    monkeypatch.setattr(
        speaker_module, "Speaker", lambda server, timeout_s: synthesizer
    )
    init_output_dir(tmp_path / "log")
    server = McpServerSpec(name="tts", url="http://127.0.0.1:1/mcp")
    toolkit = MinimalToolkit(state)
    add_frame(state)

    with narration_session(
        site, toolkit=toolkit, mcp_servers=(server,), output_dir=tmp_path
    ) as narrator:
        tools = {tool.name: tool for tool in toolkit.list_tools()}
        assert set(tools) == {"narration_start", "narration_stop"}
        assert all(tool.readonly for tool in tools.values())

        result = toolkit.execute_tool("narration_start", {"focus": "gripper"})
        assert result.data["status"] == "running"
        wait_until(lambda: len(synthesizer.texts) == 1)
        assert toolkit.execute_tool("narration_start", {"interval_s": 0.1}).is_error
        assert toolkit.execute_tool("narration_stop", {}).data["status"] == "stopping"

    assert not narrator.running
    assert describer.calls[0][1] == "gripper"
