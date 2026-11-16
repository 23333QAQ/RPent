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

"""Describe one camera frame with a vision model, independent of the planner."""

from __future__ import annotations

import base64

from rpent.narration.config import NarrationConfig

_SENTENCE_ENDS = "。！？!?；;，,"


class SceneDescriber:
    """Send each frame as a fresh, history-free chat completion request."""

    def __init__(self, config: NarrationConfig) -> None:
        from openai import AsyncOpenAI

        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_s,
            max_retries=0,
        )

    async def describe(self, image: bytes, focus: str | None = None) -> str:
        """Return one sentence describing a PNG frame.

        Args:
            image: PNG bytes of the frame.
            focus: Optional planner hint about what to emphasize.

        Raises:
            ValueError: If the model returns no text.
        """
        prompt = self._config.prompt.replace("{max_chars}", str(self._config.max_chars))
        if focus:
            prompt += f"\n重点关注：{focus}"
        data_url = "data:image/png;base64," + base64.b64encode(image).decode()
        response = await self._client.chat.completions.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        text = shorten(
            response.choices[0].message.content or "", self._config.max_chars
        )
        if not text:
            raise ValueError("vision model returned no description")
        return text

    async def close(self) -> None:
        await self._client.close()


def shorten(text: str, max_chars: int) -> str:
    """Collapse whitespace and cut overlong text at the last clause boundary."""
    text = " ".join(text.split()).strip("`*#> ")
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = max(head.rfind(mark) for mark in _SENTENCE_ENDS)
    return head[: cut + 1] if cut > 0 else head
