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
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP

from rpent.narration.config import NarrationConfig
from rpent.narration.describer import SceneDescriber, shorten
from rpent.narration.speaker import Speaker
from rpent.utils.mcp_servers import McpServerSpec


@contextmanager
def http_server(mcp: FastMCP):
    """Serve a real MCP transport on an ephemeral loopback port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(mcp.streamable_http_app(), log_level="error")
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            server.should_exit = True
            thread.join(timeout=5)


def test_speaker_returns_audio_path_and_sends_headers(monkeypatch):
    monkeypatch.setenv("TTS_TOKEN", "t0ken")
    seen = {}
    mcp = FastMCP("tts", stateless_http=True, json_response=True)

    @mcp.tool()
    async def synthesize(text: str) -> dict:
        seen["text"] = text
        return {"text": text, "audio_path": "/srv/a.wav", "mime_type": "audio/wav"}

    with http_server(mcp) as url:
        server = McpServerSpec(
            name="tts", url=url, env_http_headers={"Authorization": "TTS_TOKEN"}
        )
        path = asyncio.run(Speaker(server, timeout_s=10).synthesize("你好"))

    assert path == "/srv/a.wav"
    assert seen["text"] == "你好"


def test_speaker_surfaces_tool_errors():
    mcp = FastMCP("tts", stateless_http=True, json_response=True)

    @mcp.tool()
    async def synthesize(text: str) -> dict:
        raise ValueError("voice not installed")

    with http_server(mcp) as url:
        speaker = Speaker(McpServerSpec(name="tts", url=url), timeout_s=10)
        with pytest.raises(RuntimeError, match="voice not installed"):
            asyncio.run(speaker.synthesize("x"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  机械臂\n抓起瓶子。 ", "机械臂 抓起瓶子。"),
        ("机械臂在左侧，瓶子在右侧，篮子在后方。", "机械臂在左侧，"),
        ("一二三四五六七八九十一二三", "一二三四五六七八九十"),
        ("**桌上有杯子**", "桌上有杯子"),
    ],
)
def test_shorten_prefers_clause_boundaries(text, expected):
    assert shorten(text, 10) == expected


def test_describer_sends_a_history_free_image_request():
    config = NarrationConfig(
        tts_server="tts",
        model="vision",
        base_url="http://127.0.0.1:1/v1",
        api_key="key",
        max_chars=12,
        prompt="不超过{max_chars}字。",
    )
    describer = SceneDescriber(config)
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        message = SimpleNamespace(content="绿色瓶子在篮子旁边。")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    describer._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: asyncio.sleep(0),
    )

    text = asyncio.run(describer.describe(b"png", focus="瓶子"))

    assert text == "绿色瓶子在篮子旁边。"
    (request,) = requests
    assert request["model"] == "vision"
    (message,) = request["messages"]
    prompt, image = message["content"]
    assert prompt["text"] == "不超过12字。\n重点关注：瓶子"
    assert image["image_url"]["url"] == "data:image/png;base64,cG5n"
