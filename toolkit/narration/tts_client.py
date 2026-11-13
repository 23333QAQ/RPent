# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Consume OpenVoiceOS's MCP synthesize tool without forwarding audio to the LLM."""

import base64
import io
import json
import wave
from datetime import timedelta

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

MAX_AUDIO_BYTES = 32 * 1024 * 1024


def decode_audio(result: types.CallToolResult) -> bytes:
    """Decode the documented OVOS JSON result and require nonempty PCM WAV."""
    if result.isError:
        detail = " ".join(
            block.text
            for block in result.content
            if isinstance(block, types.TextContent)
        )
        raise RuntimeError(f"OpenVoiceOS synthesis failed: {detail[:500]}")
    payload = result.structuredContent
    if payload is None:
        blocks = [b for b in result.content if isinstance(b, types.TextContent)]
        if len(blocks) != 1:
            raise ValueError("Expected one OpenVoiceOS JSON audio result")
        payload = json.loads(blocks[0].text)
    if not isinstance(payload, dict) or payload.get("mime_type") != "audio/wav":
        raise ValueError("OpenVoiceOS must return audio/wav")
    encoded = payload.get("data")
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_AUDIO_BYTES + 2) // 3):
        raise ValueError("Missing or oversized OpenVoiceOS audio data")
    audio = base64.b64decode(encoded, validate=True)
    with wave.open(io.BytesIO(audio), "rb") as wav:
        frames = wav.getnframes()
        expected = frames * wav.getnchannels() * wav.getsampwidth()
        if not frames or len(wav.readframes(frames)) != expected:
            raise ValueError("OpenVoiceOS returned empty or truncated WAV audio")
    return audio


class OvosTtsClient:
    """Open and close one MCP session per synthesis request."""

    def __init__(
        self,
        url: str,
        *,
        voice: str | None = None,
        lang: str = "zh-CN",
        timeout: float = 120,
    ) -> None:
        self.url = url
        self.voice = voice
        self.lang = lang
        self.timeout = timeout

    async def synthesize(self, text: str) -> bytes:
        """Return WAV bytes; the remote filesystem path is not used."""
        arguments = {"text": text, "lang": self.lang}
        if self.voice:
            arguments["voice"] = self.voice
        async with httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=True
        ) as http:
            async with streamable_http_client(self.url, http_client=http) as streams:
                async with ClientSession(
                    streams[0],
                    streams[1],
                    read_timeout_seconds=timedelta(seconds=self.timeout),
                ) as session:
                    await session.initialize()
                    result = await session.call_tool("synthesize", arguments)
        return decode_audio(result)
