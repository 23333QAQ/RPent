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

"""The ``narration`` section of a site config."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpent.utils.mcp_servers import McpServerSpec

#: Bounds for the frame check interval, also enforced on planner overrides.
MIN_INTERVAL_S = 0.5
MAX_INTERVAL_S = 60.0

DEFAULT_PROMPT = (
    "用一句不超过{max_chars}个字的中文描述画面中机械臂和主要物体的状态。"
    "只输出这句话，不要使用 Markdown。"
)


@dataclass(frozen=True)
class NarrationConfig:
    """Background scene narration settings.

    Attributes:
        tts_server: Name of the ``mcp_servers`` entry exposing ``synthesize``.
        model: Vision model id for an OpenAI-compatible chat completions API.
        base_url: Base URL of that API.
        api_key: API key; use ``${oc.env:VAR}`` in the site config.
        cameras: Image artifact names in preference order; the first one
            present in the latest step is described.
        interval_s: Default seconds between checks for a new frame.
        max_chars: Upper bound on characters per spoken sentence.
        prompt: Description instruction; ``{max_chars}`` is substituted.
        max_tokens: Completion budget, including any reasoning tokens.
        request_timeout_s: Timeout for one description request.
        synthesis_timeout_s: Timeout for one speech synthesis call.
        max_queue: Descriptions waiting for synthesis before new ones drop.
        drain_timeout_s: Time allowed to synthesize queued text at shutdown.
    """

    tts_server: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    cameras: tuple[str, ...] = ("agentview_high.png", "agentview.png")
    interval_s: float = 2.0
    max_chars: int = 20
    prompt: str = DEFAULT_PROMPT
    max_tokens: int = 1024
    request_timeout_s: float = 30.0
    synthesis_timeout_s: float = 180.0
    max_queue: int = 20
    drain_timeout_s: float = 30.0


def load_narration_config(path: str | Path | None) -> NarrationConfig | None:
    """Load the ``narration`` section of a site config.

    Args:
        path: Path to the site config, or ``None`` when ``--site-config`` was
            not passed.

    Returns:
        The validated config, or ``None`` when narration is not configured.

    Raises:
        ValueError: If the site config or its ``narration`` section is invalid.
    """
    if path is None:
        return None
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ValueError(f"site config not found: {config_path}")

    from omegaconf import DictConfig, OmegaConf

    try:
        config = OmegaConf.load(config_path)
    except Exception as exc:
        raise ValueError(f"invalid site config {config_path}: {exc}") from exc
    if not isinstance(config, DictConfig):
        raise ValueError(f"site config must be a mapping: {config_path}")
    section = config.get("narration")
    if section is None:
        return None
    try:
        raw = OmegaConf.to_container(section, resolve=True)
    except Exception as exc:
        raise ValueError(f"invalid site config {config_path}: {exc}") from exc
    return parse_narration_config(raw)


def parse_narration_config(raw: Any) -> NarrationConfig:
    """Validate a ``narration`` mapping.

    Raises:
        ValueError: On unknown keys, missing required fields, or out-of-range
            values, naming the offending field.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("narration must be a mapping")
    known = set(NarrationConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"narration: unknown key(s) {unknown}. Valid keys: {sorted(known)}."
        )
    values: dict[str, Any] = {}
    for name in ("tts_server", "model", "base_url", "api_key", "prompt"):
        if name in raw:
            value = raw[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"narration.{name} must be a non-empty string")
            values[name] = value
    missing = sorted({"tts_server", "model", "base_url", "api_key"} - set(values))
    if missing:
        raise ValueError(f"narration: missing required key(s) {missing}")
    if not values["base_url"].startswith(("http://", "https://")):
        raise ValueError("narration.base_url must be an http(s) URL")
    if "cameras" in raw:
        cameras = raw["cameras"]
        if (
            isinstance(cameras, str)
            or not isinstance(cameras, list)
            or not cameras
            or not all(isinstance(name, str) and name for name in cameras)
        ):
            raise ValueError("narration.cameras must be a non-empty list of names")
        values["cameras"] = tuple(cameras)
    for name in (
        "interval_s",
        "request_timeout_s",
        "synthesis_timeout_s",
        "drain_timeout_s",
    ):
        if name in raw:
            values[name] = _positive_float(raw[name], name)
    if "interval_s" in values and not (
        MIN_INTERVAL_S <= values["interval_s"] <= MAX_INTERVAL_S
    ):
        raise ValueError(
            f"narration.interval_s must be between {MIN_INTERVAL_S:g} and "
            f"{MAX_INTERVAL_S:g} seconds"
        )
    for name in ("max_chars", "max_tokens", "max_queue"):
        if name in raw:
            value = raw[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"narration.{name} must be a positive integer")
            values[name] = value
    return NarrationConfig(**values)


def resolve_tts_server(
    config: NarrationConfig, servers: tuple[McpServerSpec, ...]
) -> McpServerSpec:
    """Return the MCP server named by ``config.tts_server``.

    Raises:
        ValueError: If no merged robot or site server has that name.
    """
    for server in servers:
        if server.name == config.tts_server:
            return server
    raise ValueError(
        f"narration.tts_server {config.tts_server!r} is not declared under mcp_servers"
    )


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"narration.{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"narration.{name} must be positive and finite")
    return value
