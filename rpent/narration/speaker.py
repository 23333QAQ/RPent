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

"""Send narration text to an external MCP ``synthesize`` tool."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from rpent.utils.mcp_servers import McpServerSpec, request_headers

TOOL_NAME = "synthesize"


class Speaker:
    """Open one MCP session per sentence and return the saved audio path."""

    def __init__(self, server: McpServerSpec, *, timeout_s: float) -> None:
        self._server = server
        self._timeout_s = timeout_s

    async def synthesize(self, text: str) -> str:
        """Synthesize ``text`` and return the ``audio_path`` the server reports.

        Raises:
            RuntimeError: If the tool reports an error.
            ValueError: If the result carries no ``audio_path``.
        """
        import httpx
        from mcp import ClientSession, types
        from mcp.client.streamable_http import streamable_http_client

        async with httpx.AsyncClient(
            headers=request_headers(self._server),
            timeout=self._timeout_s,
            follow_redirects=True,
        ) as http:
            async with streamable_http_client(self._server.url, http_client=http) as (
                read,
                write,
                *_,
            ):
                async with ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=self._timeout_s)
                ) as session:
                    await session.initialize()
                    result = await session.call_tool(TOOL_NAME, {"text": text})
        texts = [b.text for b in result.content if isinstance(b, types.TextContent)]
        if result.isError:
            raise RuntimeError(f"{TOOL_NAME} failed: {' '.join(texts)[:300]}")
        return _audio_path(result.structuredContent, texts)


def _audio_path(structured: dict[str, Any] | None, texts: list[str]) -> str:
    payload: Any = structured
    if payload is None and len(texts) == 1:
        try:
            payload = json.loads(texts[0])
        except json.JSONDecodeError:
            payload = None
    path = payload.get("audio_path") if isinstance(payload, dict) else None
    if not isinstance(path, str) or not path:
        raise ValueError(f"{TOOL_NAME} returned no audio_path")
    return path
